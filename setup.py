from setuptools import setup

f=open('requirements.txt')
req = []
for line in f:
    req.append(line.strip())
f.close()

with open("README.md", "r") as fh:
      long_description = fh.read()

setup(name='textcl',
      packages=['textcl'],
      version='1.0.1',
      license='MIT',
      description='Text preprocessing package for use in NLP tasks',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author = 'Alina Petukhova',
      author_email = 'petukhova.alina@gmail.com',
      url = 'https://github.com/alinapetukhova/textcl',
      download_url = 'https://github.com/alinapetukhova/textcl/archive/refs/tags/v.1.0.1.tar.gz',
      keywords = ['NLP', 'Text preprocessing', 'Outlier detection'],
      install_requires=req,
      python_requires='>=3.8',
      zip_safe=False,
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12'
      ])
