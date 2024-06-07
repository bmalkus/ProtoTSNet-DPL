nn(ptsnet, [TS, P])::has_proto(TS, P).

% is_class(TS, c0) :- has_proto(TS, p0).
% is_class(TS, c1) :- has_proto(TS, p1).

is_class(TS, c0) :- has_proto(TS, p0), not(has_proto(TS, p1)).
is_class(TS, c1) :- has_proto(TS, p1), not(has_proto(TS, p0)).
