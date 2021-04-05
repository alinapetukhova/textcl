from setuptools import setup

f=open('requirements.txt')
req = []
for line in f:
    req.append(line.strip())
f.close()

setup(name='textcl',
      packages=['textcl'],
      version='0.1.3',
      license='MIT',
      description='Text preprocessing package for use in NLP tasks',
      author = 'Alina Petukhova',
      author_email = 'petukhova.alina@gmail.com',
      url = 'https://github.com/alinapetukhova/textcl',
      download_url = 'https://github.com/alinapetukhova/textcl/archive/refs/tags/v.0.1.3.tar.gz',
      keywords = ['NLP', 'Text preprocessing', 'Outlier detection'],
      install_requires=req,
      python_requires='>=3.6', 
      zip_safe=False,
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8'
      ])
