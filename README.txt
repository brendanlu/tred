tred
----
tred implements a range of order-3 tensor decompositions. Mathematically, 
they rely on a novel tensor algebra introduced in [1]. In this, analogues of 
SVD, PCA, and PLS can be formulated, sharing many optimality properties with 
their matrix counterparts. 

The only top-level dependency, for an end user, is scikit-learn. We adopt many
of their utilities and design patterns so tred is very natural to anyone with 
experience with the scipy/scikit-learn stack. 

For the underlying tensor-product framework and tensor t-SVDM, see [1]. 
For the explicit rank truncation, and the TCAM algorithm, see [2]. 

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
For developer dependencies
    pip install -r requirements.txt

Please kindly run 
    black . 

In the root to autoformat code when opening pull requests to this repo.  

Credit
------
Our implementation was inspired by analogues at 
    https://github.com/scikit-learn/scikit-learn
And also by 
    https://github.com/UriaMorP/mprod_package

Future
------
We have placed a heavy emphasis on mathematical interpretability of the 
source code and computational efficiency (within the constraints of our 
dependencies) to fit within highly iterative machine-learning and statistical 
analysis workflows. 

New experimental methods may also be added to tred. Users are highly encouraged 
to visit the GitHub site, and open issues. We are very interested in optimizing 
the efficiency of the existing implementations if the need arises. 
