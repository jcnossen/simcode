import setuptools

pk = setuptools.find_packages()

setuptools.setup(
    name="smlmtorch",
    version="1.0",
    author="Jelmer Cnossen",
    author_email="j.cnossen@gmail.com",
    description="SMLM and SIMFLUX processing code, using pytorch to speed up",
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
		'numpy',
		'matplotlib',
		'tqdm',
		'tifffile',
        'h5py',
		'pyyaml',
        'lion-pytorch'
	]
)
