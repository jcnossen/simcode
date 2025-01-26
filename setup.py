import setuptools

pk = setuptools.find_packages()

setuptools.setup(
    name="smlmtorch",
    version="1.0",
    author="Jelmer Cnossen",
    author_email="j.cnossen@gmail.com",
    description="pytorch based SMLM package, including SIMCODE paper implementation",
    #long_description=long_description,
 #   long_description_content_type="text/markdown",
#    url="https",
    packages=pk,
	data_files=[],
    classifiers=[
        "Programming Language :: Python :: 3",
	    "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows"
    ],
	install_requires=[
		'numpy==1.26.4', # >2 seems to have a change in np.polyfit, breaking modulation pattern estimation (pattern_estimator.quadraticpeak)
		'matplotlib',
		'tqdm',
		'tifffile',
        'h5py',
        'scipy',
        'numba',
        'seaborn',
		'pyyaml',
        'lion-pytorch',
        'tensorboard'
	]
)
