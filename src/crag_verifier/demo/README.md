# Synthetic demo corpus

Twelve invented abstracts and twenty-four invented claims in the SciFact file format.
Every compound, protein, syndrome and finding here is made up; none of it describes real
research. The files exist so that the tests, the container smoke run and
`--backend demo` can exercise the whole pipeline without downloading SciFact.

Claims 13-18 share nothing with any abstract. Claims 19-24 are near misses in the way
many SciFact NOINFO claims are: they reuse an abstract's wording but change the
population or the object (humans instead of mice, polystyrene instead of polyethylene),
so the abstract does not decide them. Without them, acting on a wrong retrieval would
cost nothing in the demo, and the policy fitted at startup would never abstain.
