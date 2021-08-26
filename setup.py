import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="miapnlp",
    version="0.0.0",
    author="geb",
    author_email="853934146@qq.com",
    description="An efficient NLP library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MiaoDynamics/miaonlp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_data={'': ['data/*.txt']},
    install_requires=['numpy', 'pandas', 'nltk', 'tensorflow'],
    python_requires='>=3.8',
)