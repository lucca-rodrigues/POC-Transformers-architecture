from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

def main():
    # Configurar diretórios
    data_path = 'dataset/input.txt'
    output_dir = "./results"
    model_save_dir = "./fine-tune-gpt2"
    
    # Carregar dataset
    dataset = load_dataset('text', data_files={'train': data_path})
    
    # Inicializar modelo e tokenizer
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Configurar token de padding
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    
    def tokenize_function(examples):
        """Função para tokenizar os exemplos do dataset"""
        tokenized_example = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=50,
            return_tensors='pt'
        )
        return {
            'input_ids': tokenized_example['input_ids'].squeeze(),
            'attention_mask': tokenized_example['attention_mask'].squeeze()
        }
    
    # Tokenizar dataset
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    # Configurar argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=500,
        learning_rate=1e-4,
        warmup_steps=500,
        gradient_accumulation_steps=4,
        fp16=True if torch.cuda.is_available() else False,
    )
    
    # Inicializar trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )
    
    # Treinar modelo
    print("Iniciando treinamento...")
    trainer.train()
    
    # Salvar modelo e tokenizer
    print(f"Salvando modelo em {model_save_dir}")
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print("Treinamento concluído!")

if __name__ == "__main__":
    main()
