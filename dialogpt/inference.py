from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Configurar o ambiente para evitar warnings de paralelismo
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model_and_tokenizer(model_path):
    """Carrega o modelo e tokenizer treinados"""
    print("Carregando modelo e tokenizer...")
    
    # Carregar tokenizer e modelo
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Configurar tokens especiais
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = 'left'  # Importante para diálogo
    
    # Forçando uso da CPU por enquanto
    device = torch.device("cpu")
    print("Usando CPU para geração mais estável")
    
    model = model.to(device)
    return model, tokenizer, device

def generate_response(prompt, model, tokenizer, device, max_length=100):
    """Gera uma resposta para o prompt dado"""
    try:
        # Adiciona prefixo para manter o formato do treino
        formatted_prompt = f"Humano: {prompt}\nAssistente:"
        print(f"Prompt formatado: {formatted_prompt}")
        
        # Tokeniza o input com padding à esquerda
        encoded = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        print(f"Tamanho do input tokenizado: {input_ids.shape}")
        
        # Configurações de geração ajustadas para diálogo
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length + input_ids.shape[1],
            min_length=10,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )
        
        # Decodifica apenas a resposta (removendo o prompt)
        response = tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return response.strip()
    
    except Exception as e:
        print(f"Erro durante a geração: {str(e)}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."

def main():
    # Caminho para o modelo treinado
    model_path = "dialogpt/fine-tuned-model"
    
    try:
        # Carrega modelo e tokenizer
        model, tokenizer, device = load_model_and_tokenizer(model_path)
        print(f"Modelo carregado com sucesso! Usando dispositivo: {device}")
        
        print("\nDigite 'sair' para encerrar o chat\n")
        
        while True:
            user_input = input("Você: ")
            if user_input.lower() == 'sair':
                print("Encerrando chat... Até mais!")
                break
                
            response = generate_response(user_input, model, tokenizer, device)
            print(f"\nAssistente: {response}\n")
            
    except Exception as e:
        print(f"Erro ao carregar ou executar o modelo: {str(e)}")
        print("Certifique-se de que o modelo foi treinado e está no diretório correto.")

if __name__ == "__main__":
    main()
