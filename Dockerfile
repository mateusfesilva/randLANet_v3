# 1. Imagem Base: Começamos com uma imagem oficial da NVIDIA que já tem CUDA e Ubuntu.
#    Escolha a versão do CUDA compatível com o seu driver NVIDIA.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2. Evita que o instalador peça inputs interativos
ENV DEBIAN_FRONTEND=noninteractive

# 3. Instala as dependências de sistema (Linux)
#    - python3 e python3-pip para o Python
#    - build-essential para compilar C++ (g++, etc.)
#    - git, dos2unix como ferramentas úteis
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    dos2unix \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Define o diretório de trabalho dentro do container
WORKDIR /app

# 5. Copia o arquivo de requisitos e instala os pacotes Python
#    Isso aproveita o cache do Docker. Só reinstala se requirements.txt mudar.
COPY requirements-base.txt requirements.txt ./

# Instala PyTorch primeiro (comando oficial para CUDA 11.8)
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instala a base de dependências críticas PRIMEIRO
RUN pip3 install --no-cache-dir -r requirements-base.txt

# Instala o resto dos requisitos. O pip agora respeitará as versões da base já instaladas.
RUN pip3 install --no-cache-dir -r requirements.txt

# 6. Copia todo o resto do código do seu projeto para dentro do container
COPY . .

# 7. Converte os scripts .sh para o formato Unix (uma única vez)
RUN find . -type f -name "*.sh" -exec dos2unix {} \;

# 8. Executa a compilação dos operadores customizados
#    Este passo é executado UMA VEZ durante a construção da imagem.
RUN bash compile_op.sh

# 9. Define o comando padrão para quando o container iniciar (opcional)
#    Este comando mantém o container rodando para que possamos entrar nele.
CMD ["/bin/bash"]