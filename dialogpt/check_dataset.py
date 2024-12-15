import re

def check_and_fix_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_conversations = []
    current_conversation = []
    
    for line in lines:
        line = line.strip()
        if not line:  # Pular linhas vazias
            continue
            
        # Verificar se é uma linha válida (começa com Humano: ou Assistente:)
        if not (line.startswith('Humano:') or line.startswith('Assistente:')):
            print(f"Linha inválida encontrada: {line}")
            continue
            
        # Se é uma nova pergunta e temos uma conversa anterior, salvar a anterior
        if line.startswith('Humano:') and current_conversation:
            if len(current_conversation) >= 2:  # Só salvar se tiver pergunta e resposta
                fixed_conversations.extend(current_conversation)
                fixed_conversations.append('')  # Linha vazia entre conversas
            current_conversation = []
        
        # Verificar se a resposta está completa (termina com ponto final ou outro caractere de fim)
        if line.startswith('Assistente:'):
            if not re.search(r'[.!?]$', line):
                print(f"Resposta possivelmente incompleta: {line}")
                continue
        
        current_conversation.append(line)
    
    # Adicionar última conversa se existir
    if current_conversation and len(current_conversation) >= 2:
        fixed_conversations.extend(current_conversation)
    
    # Salvar dataset corrigido
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_conversations))
    
    print(f"\nEstatísticas do Dataset:")
    print(f"Total de linhas originais: {len(lines)}")
    print(f"Total de conversas válidas: {len(fixed_conversations) // 2}")
    print(f"\nDataset corrigido salvo em: {output_file}")

if __name__ == "__main__":
    input_file = '../datasets/conversations.txt'
    output_file = '../datasets/conversations_fixed.txt'
    check_and_fix_dataset(input_file, output_file)
