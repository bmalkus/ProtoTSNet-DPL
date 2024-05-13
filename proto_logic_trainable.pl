nn(ptsnet, [TS, P], H, [0, 1])::has_proto(TS, P).

t(_)::connected(p0, c0).
t(_)::connected(p1, c1).

is_class(TS, c0) :- has_proto(TS, p0), connected(p0, c0).
is_class(TS, c1) :- has_proto(TS, p1), connected(p1, c1).
