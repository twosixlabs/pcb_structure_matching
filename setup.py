from setuptools import setup

def get_reqs(reqs_path):
    with open(reqs_path) as f:
        return [line for line in f.readlines() if not line.startswith("#")]

setup(
    name='pcb_comp_detector',
    version='0.1.0',
    description='PCB Component Detector using detectron2 and associated capabilities',
    url='https://gitlab.com/repo-rebase/pcb_comp_detector',
    author='Calvin Hirsch',
    author_email='calvin.hirsch@twosixtech.com',
    license='BSD 2-clause',
    install_requires=get_reqs("requirements/requirements_detectron2.txt") +
                     get_reqs("requirements/requirements.txt"),
    extras_require={
        "scripts": get_reqs("requirements/requirements_scripts.txt")
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3',
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)