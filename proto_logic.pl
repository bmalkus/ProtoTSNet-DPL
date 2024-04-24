nn(ptsnet, [TS, P], H, [0])::has_proto(TS, P, H).

is_class(TS, c0) :- class(c0), has_proto(TS, p0).
is_class(TS, c1) :- class(c1), has_proto(TS, p1).

class(c0).
class(c1).
