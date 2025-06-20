from setuptools import setup, find_packages

setup(
    name='ci-library',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for recording, vectorizing, and analyzing cochlear implant electrodograms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/ci-library',  # optional
    packages=find_packages(include=['ci_processor', 'ci_processor.*']),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'soundfile',
        'sounddevice',
        'pandas',
        'matplotlib',
        'streamlit',
        'plotly',           # if used in streamlit_app
        'zarr',             # if you're saving output to .zarr
    ],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
    entry_points={
        'console_scripts': [
            'ci-vectorize = ci_processor.main_pipeline:main',  # Optional CLI entry point
        ]
    }
)
