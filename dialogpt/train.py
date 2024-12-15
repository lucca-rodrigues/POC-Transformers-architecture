from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

def main():
    # Configurar diretórios
    data_path = 'datasets/conversations.txt'
    output_dir = "dialogpt/results"
    model_save_dir = "dialogpt/fine-tuned-model"
    
    # Carregar dataset
    dataset = load_dataset('text', data_files={'train': data_path})
    
    # Inicializar modelo e tokenizer
    model_name = 'microsoft/DialoGPT-small'  # Podemos usar também medium ou large
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Configurar tokens especiais
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    def tokenize_conversations(examples):
        """
        Tokeniza as conversas. Formato esperado no arquivo:
        Humano: Olá, como vai você?
        Assistente: Oi! Estou bem, obrigado por perguntar.
        """
        max_length = 128
        inputs = []
        
        for text in examples['text']:
            # Processa o texto e adiciona tokens especiais
            conversation = text.strip()
            encoded = tokenizer.encode(conversation, truncation=True, max_length=max_length)
            inputs.append(encoded)
        
        # Padding
        max_len = max(len(x) for x in inputs)
        padded_inputs = []
        attention_masks = []
        
        for encoded in inputs:
            padding_length = max_len - len(encoded)
            padded_inputs.append(encoded + [tokenizer.pad_token_id] * padding_length)
            attention_masks.append([1] * len(encoded) + [0] * padding_length)
            
        return {
            'input_ids': padded_inputs,
            'attention_mask': attention_masks,
            'labels': padded_inputs.copy()  # Para treinamento autoregressivo
        }

    # Tokenizar dataset
    tokenized_datasets = dataset.map(
        tokenize_conversations,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Configurar formato para PyTorch
    tokenized_datasets.set_format(type='torch')
    
    # Configurar argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
        warmup_steps=100,
        gradient_accumulation_steps=4,
        fp16=torch.cuda.is_available(),
    )
    
    # Inicializar trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )
    
    # Treinar modelo
    print("Iniciando treinamento do DialoGPT...")
    trainer.train()
    
    # Salvar modelo e tokenizer
    print(f"Salvando modelo em {model_save_dir}")
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print("Treinamento concluído!")

if __name__ == "__main__":
    main()
