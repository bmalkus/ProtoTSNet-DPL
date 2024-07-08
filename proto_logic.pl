nn(ptsnet, [TS, P])::has_proto(TS, P).

t(0.75)::connected(p0, c0).
t(0.75)::connected(p1, c1).
t(0.75)::connected(p2, c2).
t(0.75)::connected(p3, c3).

% t(_)::connected(p0, c1).

is_class(TS, c0) :- has_proto(TS, p0), connected(p0, c0).
is_class(TS, c1) :- has_proto(TS, p1), connected(p1, c1).
is_class(TS, c2) :- has_proto(TS, p2), connected(p2, c2).
is_class(TS, c3) :- has_proto(TS, p3), connected(p3, c3).
% is_class(TS, c2) :- has_proto(TS, p2).
