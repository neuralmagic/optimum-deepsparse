from setuptools import find_namespace_packages, setup

INSTALL_REQUIRE = [
    "deepsparse-nightly", # TODO: move to stable release
    "optimum>=1.8.0",
    "transformers>=4.20.0",
    "datasets>=1.4.0",
    "sentencepiece",
    "scipy",
]

TESTS_REQUIRE = ["pytest", "parameterized", "Pillow", "evaluate", "diffusers", "py-cpuinfo"]

QUALITY_REQUIRE = ["black~=23.1", "ruff>=0.0.241"]

EXTRA_REQUIRE = {
    "testing": [
        "filelock",
        "GitPython",
        "parameterized",
        "psutil",
        "pytest",
        "pytest-pythonpath",
        "pytest-xdist",
        "librosa",
        "soundfile",
    ],
    "quality": QUALITY_REQUIRE,
}

setup(
    name="optimum-deepsparse",
    version=__version__,
    description="Optimum DeepSparse is an extension of the Hugging Face Transformers library "
    "that integrates the DeepSparse inference runtime. DeepSparse offers GPU-class performance "
    "on CPUs, making it possible to run Transformers and other deep learning models on commodity "
    "hardware with sparsity. Optimum DeepSparse provides a framework for developers to easily "
    "integrate DeepSparse into their applications, regardless of the hardware platform.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="inference, cpu, x86, arm, transformers, quantization, pruning, sparsity",
    url="https://github.com/neuralmagic/deepsparse",
    author="Neuralmagic, Inc.",
    author_email="michael@neuralmagic.com",
    license="Neural Magic DeepSparse Community License, Apache",
    packages=find_namespace_packages(include=["optimum*"]),
    install_requires=INSTALL_REQUIRE,
    extras_require=EXTRA_REQUIRE,
    python_requires=">=3.8, <3.11",
    include_package_data=True,
    zip_safe=False,
    entry_points={"console_scripts": ["optimum-cli=optimum.commands.optimum_cli:main"]},
)
