from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator

def main():
    # Configurar diretórios
    data_path = 'datasets/conversations_fixed.txt'  # Corrigindo o caminho
    output_dir = "dialogpt/results"
    model_save_dir = "dialogpt/fine-tuned-model"
    
    # Verificar disponibilidade de aceleração de hardware
    print("Verificando disponibilidade de aceleração de hardware...")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Usando GPU (CUDA)")
        # Informações da GPU
        print(f"GPU disponível: {torch.cuda.get_device_name(0)}")
        print(f"Memória GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Usando aceleração M1/M2 (MPS)")
        # Teste MPS
        test_tensor = torch.randn(2, 2).to(device)
        print(f"Teste MPS - Device do tensor: {test_tensor.device}")
    else:
        device = torch.device("cpu")
        print("Usando CPU")
        print("Motivo: Nenhuma aceleração de hardware (CUDA ou MPS) disponível")
    
    # Configurar Accelerator para otimização
    accelerator = Accelerator()
    
    # Carregar dataset
    print("Carregando dataset...")
    dataset = load_dataset('text', data_files={'train': data_path})
    
    # Inicializar modelo e tokenizer
    print("Inicializando modelo e tokenizer...")
    model_name = 'microsoft/DialoGPT-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Mover modelo para o dispositivo apropriado
    model = model.to(device)
    
    # Configurar tokens especiais
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = 'left'  # Importante para diálogo
    
    def tokenize_conversations(examples):
        """
        Tokeniza as conversas com formato adequado para diálogo
        """
        max_length = 128
        
        # Processar todas as conversas do batch
        conversations = []
        for text in examples['text']:
            # Garantir que cada conversa começa com 'Humano:' e tem 'Assistente:'
            if not text.startswith('Humano:'):
                text = 'Humano: ' + text
            if 'Assistente:' not in text:
                continue
            conversations.append(text.strip())
        
        # Tokenizar com padding e truncamento
        tokenized = tokenizer(
            conversations,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        # Preparar labels
        labels = tokenized['input_ids'].clone()
        
        # Mascarar tokens de padding nos labels
        labels[labels == tokenizer.pad_token_id] = -100
        
        # Mascarar a parte do input (apenas treinar na resposta)
        for i, conv in enumerate(conversations):
            human_part = conv.split('Assistente:')[0]
            human_tokens = len(tokenizer.encode(human_part))
            labels[i, :human_tokens] = -100
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }

    # Tokenizar dataset com processamento em lotes otimizado
    print("Tokenizando dataset...")
    tokenized_datasets = dataset.map(
        tokenize_conversations,
        batched=True,
        batch_size=32,  # Reduzido para melhor gestão de memória
        remove_columns=dataset["train"].column_names,
        desc="Tokenizando conversas"
    )
    
    # Configurar formato para PyTorch
    tokenized_datasets.set_format(type='torch')
    
    # Criar data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Configurar TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # Aumentado para melhor aprendizado
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=True,
        learning_rate=5e-5,  # Taxa de aprendizado ajustada
    )
    
    # Criar Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
    )
    
    # Treinar modelo
    print("Iniciando treinamento...")
    trainer.train()
    
    # Salvar modelo treinado
    print(f"\nSalvando modelo em {model_save_dir}")
    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    
    print("Treinamento concluído!")

if __name__ == "__main__":
    main()
