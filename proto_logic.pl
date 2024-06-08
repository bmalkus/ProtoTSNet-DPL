nn(ptsnet, [TS, P])::has_proto(TS, P).

% is_class(TS, c0) :- has_proto(TS, p0).
% is_class(TS, c1) :- has_proto(TS, p1).

t(0.5)::half_has_proto(TS, P) :- has_proto(TS, P).

is_class(TS, c0) :- has_proto(TS, p0), not(half_has_proto(TS, p1)).
is_class(TS, c1) :- has_proto(TS, p1), not(half_has_proto(TS, p0)).
