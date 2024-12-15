from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_and_tokenizer(model_path):
    """Carrega o modelo e tokenizer treinados"""
    print("Carregando modelo e tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Mover para GPU se disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, tokenizer, device

def generate_response(prompt, model, tokenizer, device, max_length=100):
    """Gera uma resposta para o prompt dado"""
    # Adiciona prefixo para manter o formato do treino
    formatted_prompt = f"Humano: {prompt}\nAssistente:"
    
    # Tokeniza o input
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
    
    # Configurações de geração
    attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=device)
    pad_token_id = tokenizer.eos_token_id
    
    # Gera resposta
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length + inputs.shape[1],
        temperature=0.7,
        num_return_sequences=1,
        pad_token_id=pad_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2
    )
    
    # Decodifica a resposta
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    # Caminho para o modelo treinado
    model_path = "./fine-tune-dialogpt"
    
    try:
        # Carrega modelo e tokenizer
        model, tokenizer, device = load_model_and_tokenizer(model_path)
        print(f"Modelo carregado com sucesso! Usando dispositivo: {device}")
        print("\nDigite 'sair' para encerrar o chat")
        
        # Loop de conversação
        while True:
            # Obtém input do usuário
            user_input = input("\nVocê: ")
            
            if user_input.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando chat... Até mais!")
                break
            
            # Gera e mostra resposta
            print("\nAssistente: ", end="")
            response = generate_response(user_input, model, tokenizer, device)
            print(response)

    except Exception as e:
        print(f"Erro ao carregar ou executar o modelo: {str(e)}")
        print("Certifique-se de que o modelo foi treinado e está no diretório correto.")

if __name__ == "__main__":
    main()
