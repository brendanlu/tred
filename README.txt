tred
----
tred implements a range of order-3 tensor decompositions. Mathematically, 
they rely on a novel tensor algebra introduced in [1]. Within this framework, 
natural tensor analogues of SVD and PCA were recently formulated; these 
decomposition techniques also have analytically proven optimality properties 
that mirror those of their matrix counterparts. 

The only top-level dependency, for an end user, is scikit-learn. We inherit from 
their base classes, so tred's class API's should be natural to any past 
scikit-learn users. tred's function API's also mirror scipy counterparts as much
as possible. 

For the underlying tensor-product framework and tensor t-SVDM, see [1]. 
For the explicit rank truncation, and the TCAM algorithm, see [2]. 

Use
---
[MORE USER-FRIENDLY EXAMPLES WILL BE UP SOON]

tred is on PyPI, simply run
        pip install --upgrade tred
To use the package. 

tred is in active development. But, most of the existing implementations are 
covered by unit tests, and we aim to keep the API relatively stable. 

Users are highly encouraged to visit the GitHub site, and open issues.

Literature
----------
[1] Kilmer, M.E., Horesh, L., Avron, H. and Newman, E., 2021. Tensor-tensor 
algebra for optimal representation and compression of multiway data. Proceedings 
of the National Academy of Sciences, 118(28), p.e2015851118.

[2] Mor, U., Cohen, Y., Valdés-Mas, R., Kviatcovsky, D., Elinav, E. and Avron, 
H., 2022. Dimensionality reduction of longitudinal’omics data using modern 
tensor factorizations. PLoS Computational Biology, 18(7), p.e1010212.

NOTE: In literature, the authors use m, p, n as the dimensions of the tensors, 
whereas throughout this package one will see we prefer n, p, t instead. We will 
also use k = min(n, p), where from an `omics analysis perspective, typically 
means k = n, as p >> n typically. 

Development
-----------
For anyone who is interested in adding to the package:

    For development dependencies, invoke:
            pip install -r requirements.txt
    In the root to install all of the required packages for development. 

    To test any changes, invoke:
            pytest . 
    In the root to run the tests in the test folder.

    If opening pull requests, invoke:
            black . 
    In the root to autoformat code. 

    To build an updated version of the docs, invoke:
            pdoc --html --output-dir docs tred
    In the root to generate the docs in html under a `docs` directory. 

Much of the implementation and code practice mirrors that of scikit-learn. We
adopt their utilities and general coding guidelines whenever we can. 

For any interested contributors, please be very liberal in adding tests, both 
for new and existing features. 

Credit
------
Our implementation was inspired by analogues at 
    https://github.com/scikit-learn/scikit-learn
And also by 
    https://github.com/UriaMorP/mprod_package

Future
------
New experimental methods may be added to tred. 
