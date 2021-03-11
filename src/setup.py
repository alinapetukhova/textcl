from setuptools import setup

f=open('requirements.txt')
req = []
for line in f:
    req.append(line.strip())
f.close()
    

setup(name='textcl',
      version='0.1.0',
      description='Package for text preprocessing to use in nlp tasks',
      packages=['textcl'],
      install_requires=req,
      python_requires='>=3.6', 
      zip_safe=False)
