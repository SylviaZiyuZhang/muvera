from setuptools import setup

setup(
        name = 'muvera_pybind',
        version  = '0.0.1',
        author = '',
        packages   = ['muvera_pybind'],
        include_package_data = True,
        package_data = { 'muvera_pybind': ['*.so'] },
        zip_safe  = False
)