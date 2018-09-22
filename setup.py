import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='sigpy',
    version='0.1',
    author='Riley John Murray',
    author_email='rmurray@caltech.edu',
    description='Signomial optimiziation via SAGE.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rileyjmurray/sigpy',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python',
        'Natural Language :: English',
        'License :: Apache Software License',
        'Development Status :: 3 - Alpha',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    install_requires=["coniclifts >= 0.1",
                      "ecos >= 2",
                      "numpy >= 1.14"
    ]
)
