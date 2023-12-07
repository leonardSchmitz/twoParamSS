J. Diehl, L. Schmitz, 2022.
*Two-parameter sums signatures and corresponding quasisymmetric functions*. arXiv preprint arXiv:2210.14247.
[https://arxiv.org/abs/2210.14247]


The 2-parameter sums signatures is implemented in `twoParameterSS.py` for arbitrary dimension d. 
The matrix composition Hopf algebra is implemented in composition_Algebra.py for arbitrary d. 
"picturesLoding.py" is a small library for image preperation based on warping. 
Sample pictures are provided in the folder "pictures". 

The code uses a library for linear combinations by J. Diehl:
[https://github.com/diehlj/linear-combination-py]

tests.py contains numeric tests for the article, including 
- Example 2.5. (toy computation SS)
- Example 2.7. (toy computation quasi-shuffle) 
- Example 2.9  (toy computation coproduct)
- Theorem 2.10. (Hopf algebra)
- Theorem 2.11. (Quasi-shuffle identity)    
- Example 2.13. (toy computation quasi-shuffle)
- Lemma 2.15. (Zero properties)
- Theorem 2.18. (SS zero insertion invariant)
- Lemma 2.19. (warp properties)
- Lemma 2.20. (delta, sigma properties)
- Lemma 2.22. (identeties relating delta, Zero, warp, normal forms)
- Theorem 2.30 (SS(delta(.)) invariant to warping)
- Example 2.32 (toy computation diagonal concatenation in evZ)
- Lemma 2.33 (Chen's identity in evZ)
- Example 2.35 (toy computation diagonal concatenaion in evC)
- Theorem 4.5 (efficient computation of SS w.r.t. chained 1x1 compositions)
- Lemma 5.6 (delta semi-group homomorphism)
- Lemma 5.7 (sigma semi-group homomorphism)


The test file also uses pictures to compute the 2-parameter signatues with d=3. 

