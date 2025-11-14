# Importa as ferramentas de construção modernas
from setuptools import setup, Extension
import numpy

# Define a extensão C++ a ser compilada
ext_modules = [
    Extension(
        # Nome do módulo final
        'grid_subsampling',
        
        # Lista de todos os arquivos de código-fonte C++ necessários
        sources=[
            'wrapper.cpp',
            'grid_subsampling/grid_subsampling.cpp',
            '../cpp_utils/cloud/cloud.cpp'
        ],
        
        # Argumentos extras para o compilador
        extra_compile_args=['-std=c++11', '-D_GLIBCXX_USE_CXX11_ABI=0'],
        
        # Informa ao compilador onde encontrar os arquivos de cabeçalho do NumPy
        include_dirs=[numpy.get_include()]
    )
]

# Executa a configuração
setup(
    name='grid_subsampling',
    ext_modules=ext_modules,
)