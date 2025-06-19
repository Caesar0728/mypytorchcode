from modelonlinefinal import build_transformer
from datasetonlinefinal import BilingualDataset, look_ahead_mask
from configuration import get_config, get_weights_file_path, latest_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def get_all_sentences(dataset, language):

    for item in dataset:

        yield item['translation'][language]


def get_or_build_tokenizer(configuration, dataset, language):

    tokenizer_path = Path(configuration['tokenizer_file'].format(language))

    if not Path.exists(tokenizer_path):

        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    else:

        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(configuration):

    # It only has the train split, so we divide it overselves
    dataset_raw = load_dataset(path=f"{configuration['datasource']}",
                               name=f"{configuration['lang_src']}-{configuration['lang_tgt']}",
                               split='train')
    # need to fill in XXXXX for the data path
    # data_files = f"D:/Users/XXXXX/{configuration['datasource']}/{configuration['lang_src']}-{configuration['lang_tgt']}/*"
    # dataset_raw = load_dataset("parquet", data_files=data_files, split='train')


    # Keep 90% for training, 10% for validation
    train_dataset_size = int(0.9 * len(dataset_raw))
    val_dataset_size = len(dataset_raw) - train_dataset_size
    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_dataset_size, val_dataset_size])

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(configuration, dataset_raw, configuration['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(configuration, dataset_raw, configuration['lang_tgt'])

    train_dataset = BilingualDataset(train_dataset_raw,
                                     tokenizer_src,
                                     tokenizer_tgt,
                                     configuration['lang_src'],
                                     configuration['lang_tgt'],
                                     configuration['seq_len'])

    val_dataset = BilingualDataset(val_dataset_raw,
                                   tokenizer_src,
                                   tokenizer_tgt,
                                   configuration['lang_src'],
                                   configuration['lang_tgt'],
                                   configuration['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in dataset_raw:

        src_ids = tokenizer_src.encode(item['translation'][configuration['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][configuration['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_dataset, batch_size=configuration['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(configuration, src_vocab_size, tgt_vocab_size):

    model = build_transformer(src_vocab_size,
                              tgt_vocab_size,
                              configuration['seq_len'],
                              configuration['seq_len'],
                              configuration['d_model'])
                                # d_model=configuration['d_model'])
    return model


def greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device):

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(encoder_input, encoder_mask)

    tgt_sos_token = tokenizer_tgt.token_to_id('[SOS]')
    tgt_eos_token = tokenizer_tgt.token_to_id('[EOS]')

    # Initialize the decoder input with the sos token
    # decoder_input = torch.empty(1, 1).fill_(tgt_sos_token).type_as(encoder_input).to(device)
    decoder_input = torch.tensor(tgt_sos_token).view(1, 1).type_as(encoder_input).to(device)

    # step 4 while loop
    while True:

        if decoder_input.size(1) == max_len:

            break

        # build mask for target
        decoder_mask = look_ahead_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

        # calculate output
        decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)

        # get next token
        prob = model.project(decoder_output[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.tensor(next_word.item()).view(1, 1).type_as(encoder_input).to(device)
            ], dim=1
             # torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == tgt_eos_token:
            break
    decoder_input = decoder_input.squeeze(0)

    return decoder_input


def run_validation(model, validation_dataset, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, num_examples=2):

    model.eval()
    count = 0
    console_width = 80

    with torch.no_grad():

        for batch in validation_dataset:

            count += 1

            encoder_input = batch['encoder_input'].to(device)  # (batch_size, seq_len)

            encoder_mask = batch['encoder_mask'].to(device)  # (batch_size, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_output = greedy_decode(model,
                                         encoder_input,
                                         encoder_mask,
                                         tokenizer_src,
                                         tokenizer_tgt,
                                         max_len,
                                         device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            # Print the source, target and model output
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_output_text}")

            if count == num_examples:

                print_msg('-' * console_width)

                break


def train_model(configuration):

    # step 1 define the device
    # device = "cuda" if torch.cuda.is_available() else "cpu"  #  else "mps" if torch.has_mps or torch.backends.mps.is_available()
    # print("Using device:", device)
    # if (device == 'cuda'):
    #     print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    #     print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    # elif (device == 'mps'):
    #     print(f"Device name: <mps>")
    # else:
    #     print("NOTE: If you have a GPU, consider using it for training.")
    #     print(
    #         "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
    #     print(
    #         "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    # device = torch.device(device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                          # else 'mps' if torch.backends.mps.is_available()

    print(f"Using device {device}")

    # Make sure the weights folder exists
    # here we need to have the smae path name as in configuration class, they have to align with each other
    # {configuration['datasource']}_{configuration['model_folder']}
    Path(f"{configuration['datasource']}_{configuration['model_folder']}").mkdir(parents=True, exist_ok=True)

    # step 3 get_ds
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(configuration)

    # step 4 get_model
    model = get_model(configuration, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(configuration['experiment_name'])

    # step 5 criterion
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    # step 6 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['lr'], eps=1e-9)

    # step 7 preload
    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    # preload = configuration['preload']
    # model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config,
    #                                                                                                     preload) if preload else None
    # if model_filename:
    if configuration['preload']:

        model_filename = get_weights_file_path(configuration, configuration['preload'])
        print(f"Preloading model: {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    else:

        print("No model to preload, starting from scratch!")

    # step 8 looping
    for epoch in range(initial_epoch, configuration['num_epochs']):

        # torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch: {epoch:02d}")

        for batch in batch_iterator:

            # print(f" this is batch shape {batch.shape}")
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)

            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            proj_output = proj_output.view(-1, tokenizer_tgt.get_vocab_size())
            # Compare the output with the label

            label = batch['label'].to(device)  # (B, seq_len)
            label = label.view(-1)
            # Compute the loss using a simple cross entropy
            loss = criterion(proj_output, label)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model,
                       val_dataloader,
                       tokenizer_src,
                       tokenizer_tgt,
                       configuration['seq_len'],
                       device,
                       print_msg=lambda msg: batch_iterator.write(msg),
                       num_examples=2)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(configuration, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step
            }, model_filename
        )


if __name__ == '__main__':

    configuration = get_config()
    train_model(configuration)
