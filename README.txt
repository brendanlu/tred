TRED
----
Credit inspiration for our package: https://github.com/UriaMorP/mprod_package

For the underlying tensor-product framework and tensor t-SVDM, see [1]. 

For the explicit rank truncation, and the TCAM algorithm, see [2]. 

References: 
[1] Kilmer, M.E., Horesh, L., Avron, H. and Newman, E., 2021. Tensor-tensor 
algebra for optimal representation and compression of multiway data. Proceedings 
of the National Academy of Sciences, 118(28), p.e2015851118.

[2] Mor, U., Cohen, Y., Valdés-Mas, R., Kviatcovsky, D., Elinav, E. and Avron, 
H., 2022. Dimensionality reduction of longitudinal’omics data using modern 
tensor factorizations. PLoS Computational Biology, 18(7), p.e1010212.

NOTE: The original authors use m, p, n as the dimensions of the tensors, whereas 
throughout this package one will see we prefer n, p, t instead. We will also use
k = min(n, p), where from an `omics analysis perspective, typically means that
k = n. 