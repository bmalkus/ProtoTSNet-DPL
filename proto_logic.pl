nn(ptsnet, [TS, P])::has_proto(TS, P).

and(A, B) :- A, B.

t(5)::connected(p0, c0).
t(5)::connected(p1, c1).
t(5)::connected(p2, c2).
t(5)::connected(p3, c3).
t(5)::connected(p4, c4).
t(5)::connected(p5, c5).
t(5)::connected(p6, c6).
t(5)::connected(p7, c7).
t(5)::connected(p8, c8).
t(5)::connected(p9, c9).
t(5)::connected(p10, c10).
t(5)::connected(p11, c11).
t(5)::connected(p12, c12).
t(5)::connected(p13, c13).
t(5)::connected(p14, c14).

is_class(TS, c0) :- has_proto(TS, p0), connected(p0, c0).
is_class(TS, c1) :- has_proto(TS, p1), connected(p1, c1).
is_class(TS, c2) :- has_proto(TS, p2), connected(p2, c2).
is_class(TS, c3) :- has_proto(TS, p3), connected(p3, c3).
is_class(TS, c4) :- has_proto(TS, p4), connected(p4, c4).
is_class(TS, c5) :- has_proto(TS, p5), connected(p5, c5).
is_class(TS, c6) :- has_proto(TS, p6), connected(p6, c6).
is_class(TS, c7) :- has_proto(TS, p7), connected(p7, c7).
is_class(TS, c8) :- has_proto(TS, p8), connected(p8, c8).
is_class(TS, c9) :- has_proto(TS, p9), connected(p9, c9).
is_class(TS, c10) :- has_proto(TS, p10), connected(p10, c10).
is_class(TS, c11) :- has_proto(TS, p11), connected(p11, c11).
is_class(TS, c12) :- has_proto(TS, p12), connected(p12, c12).
is_class(TS, c13) :- has_proto(TS, p13), connected(p13, c13).
is_class(TS, c14) :- has_proto(TS, p14), connected(p14, c14).

excl_is_class(TS, c0) :- is_class(TS, c0), not(is_class(TS, c1)), not(is_class(TS, c2)), not(is_class(TS, c3)), not(is_class(TS, c4)), not(is_class(TS, c5)), not(is_class(TS, c6)), not(is_class(TS, c7)), not(is_class(TS, c8)), not(is_class(TS, c9)), not(is_class(TS, c10)), not(is_class(TS, c11)), not(is_class(TS, c12)), not(is_class(TS, c13)), not(is_class(TS, c14)).
excl_is_class(TS, c1) :- not(is_class(TS, c0)), is_class(TS, c1), not(is_class(TS, c2)), not(is_class(TS, c3)), not(is_class(TS, c4)), not(is_class(TS, c5)), not(is_class(TS, c6)), not(is_class(TS, c7)), not(is_class(TS, c8)), not(is_class(TS, c9)), not(is_class(TS, c10)), not(is_class(TS, c11)), not(is_class(TS, c12)), not(is_class(TS, c13)), not(is_class(TS, c14)).
excl_is_class(TS, c2) :- not(is_class(TS, c0)), not(is_class(TS, c1)), is_class(TS, c2), not(is_class(TS, c3)), not(is_class(TS, c4)), not(is_class(TS, c5)), not(is_class(TS, c6)), not(is_class(TS, c7)), not(is_class(TS, c8)), not(is_class(TS, c9)), not(is_class(TS, c10)), not(is_class(TS, c11)), not(is_class(TS, c12)), not(is_class(TS, c13)), not(is_class(TS, c14)).
excl_is_class(TS, c3) :- not(is_class(TS, c0)), not(is_class(TS, c1)), not(is_class(TS, c2)), is_class(TS, c3), not(is_class(TS, c4)), not(is_class(TS, c5)), not(is_class(TS, c6)), not(is_class(TS, c7)), not(is_class(TS, c8)), not(is_class(TS, c9)), not(is_class(TS, c10)), not(is_class(TS, c11)), not(is_class(TS, c12)), not(is_class(TS, c13)), not(is_class(TS, c14)).
excl_is_class(TS, c4) :- not(is_class(TS, c0)), not(is_class(TS, c1)), not(is_class(TS, c2)), not(is_class(TS, c3)), is_class(TS, c4), not(is_class(TS, c5)), not(is_class(TS, c6)), not(is_class(TS, c7)), not(is_class(TS, c8)), not(is_class(TS, c9)), not(is_class(TS, c10)), not(is_class(TS, c11)), not(is_class(TS, c12)), not(is_class(TS, c13)), not(is_class(TS, c14)).
excl_is_class(TS, c5) :- not(is_class(TS, c0)), not(is_class(TS, c1)), not(is_class(TS, c2)), not(is_class(TS, c3)), not(is_class(TS, c4)), is_class(TS, c5), not(is_class(TS, c6)), not(is_class(TS, c7)), not(is_class(TS, c8)), not(is_class(TS, c9)), not(is_class(TS, c10)), not(is_class(TS, c11)), not(is_class(TS, c12)), not(is_class(TS, c13)), not(is_class(TS, c14)).
excl_is_class(TS, c6) :- not(is_class(TS, c0)), not(is_class(TS, c1)), not(is_class(TS, c2)), not(is_class(TS, c3)), not(is_class(TS, c4)), not(is_class(TS, c5)), is_class(TS, c6), not(is_class(TS, c7)), not(is_class(TS, c8)), not(is_class(TS, c9)), not(is_class(TS, c10)), not(is_class(TS, c11)), not(is_class(TS, c12)), not(is_class(TS, c13)), not(is_class(TS, c14)).
excl_is_class(TS, c7) :- not(is_class(TS, c0)), not(is_class(TS, c1)), not(is_class(TS, c2)), not(is_class(TS, c3)), not(is_class(TS, c4)), not(is_class(TS, c5)), not(is_class(TS, c6)), is_class(TS, c7), not(is_class(TS, c8)), not(is_class(TS, c9)), not(is_class(TS, c10)), not(is_class(TS, c11)), not(is_class(TS, c12)), not(is_class(TS, c13)), not(is_class(TS, c14)).
excl_is_class(TS, c8) :- not(is_class(TS, c0)), not(is_class(TS, c1)), not(is_class(TS, c2)), not(is_class(TS, c3)), not(is_class(TS, c4)), not(is_class(TS, c5)), not(is_class(TS, c6)), not(is_class(TS, c7)), is_class(TS, c8), not(is_class(TS, c9)), not(is_class(TS, c10)), not(is_class(TS, c11)), not(is_class(TS, c12)), not(is_class(TS, c13)), not(is_class(TS, c14)).
excl_is_class(TS, c9) :- not(is_class(TS, c0)), not(is_class(TS, c1)), not(is_class(TS, c2)), not(is_class(TS, c3)), not(is_class(TS, c4)), not(is_class(TS, c5)), not(is_class(TS, c6)), not(is_class(TS, c7)), not(is_class(TS, c8)), is_class(TS, c9), not(is_class(TS, c10)), not(is_class(TS, c11)), not(is_class(TS, c12)), not(is_class(TS, c13)), not(is_class(TS, c14)).
excl_is_class(TS, c10) :- not(is_class(TS, c0)), not(is_class(TS, c1)), not(is_class(TS, c2)), not(is_class(TS, c3)), not(is_class(TS, c4)), not(is_class(TS, c5)), not(is_class(TS, c6)), not(is_class(TS, c7)), not(is_class(TS, c8)), not(is_class(TS, c9)), is_class(TS, c10), not(is_class(TS, c11)), not(is_class(TS, c12)), not(is_class(TS, c13)), not(is_class(TS, c14)).
excl_is_class(TS, c11) :- not(is_class(TS, c0)), not(is_class(TS, c1)), not(is_class(TS, c2)), not(is_class(TS, c3)), not(is_class(TS, c4)), not(is_class(TS, c5)), not(is_class(TS, c6)), not(is_class(TS, c7)), not(is_class(TS, c8)), not(is_class(TS, c9)), not(is_class(TS, c10)), is_class(TS, c11), not(is_class(TS, c12)), not(is_class(TS, c13)), not(is_class(TS, c14)).
excl_is_class(TS, c12) :- not(is_class(TS, c0)), not(is_class(TS, c1)), not(is_class(TS, c2)), not(is_class(TS, c3)), not(is_class(TS, c4)), not(is_class(TS, c5)), not(is_class(TS, c6)), not(is_class(TS, c7)), not(is_class(TS, c8)), not(is_class(TS, c9)), not(is_class(TS, c10)), not(is_class(TS, c11)), is_class(TS, c12), not(is_class(TS, c13)), not(is_class(TS, c14)).
excl_is_class(TS, c13) :- not(is_class(TS, c0)), not(is_class(TS, c1)), not(is_class(TS, c2)), not(is_class(TS, c3)), not(is_class(TS, c4)), not(is_class(TS, c5)), not(is_class(TS, c6)), not(is_class(TS, c7)), not(is_class(TS, c8)), not(is_class(TS, c9)), not(is_class(TS, c10)), not(is_class(TS, c11)), not(is_class(TS, c12)), is_class(TS, c13), not(is_class(TS, c14)).
excl_is_class(TS, c14) :- not(is_class(TS, c0)), not(is_class(TS, c1)), not(is_class(TS, c2)), not(is_class(TS, c3)), not(is_class(TS, c4)), not(is_class(TS, c5)), not(is_class(TS, c6)), not(is_class(TS, c7)), not(is_class(TS, c8)), not(is_class(TS, c9)), not(is_class(TS, c10)), not(is_class(TS, c11)), not(is_class(TS, c12)), not(is_class(TS, c13)), is_class(TS, c14).
