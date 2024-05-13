% nn(ptsnet, [TS, P], H, [1])::has_proto(TS, P).
nn(ptsnet, [TS], P, [p0, p1])::has_proto(TS, P).
% p::has_proto(TS, P, 1); (1-p)::has_proto(TS, P, 0).
% Uczenie sieci z warstwą gestą i spięcie z DPL
% sprawdzić prawdopodobieństwa w czasie uczenia
% sprawdzić prototypy jako AD

0.5::half_has_not_proto(TS, P) :- \+ has_proto(TS, P).

% is_class(TS, c0) :- has_proto(TS, p0), \+ half_has_proto(TS, p1).
% is_class(TS, c1) :- has_proto(TS, p1), \+ half_has_proto(TS, p0).

is_class(TS, c0) :- has_proto(TS, p0).
is_class(TS, c1) :- has_proto(TS, p1).

% is_class(TS, c0) :- has_proto(TS, p0), half_has_not_proto(TS, p1).
% is_class(TS, c1) :- has_proto(TS, p1), half_has_not_proto(TS, p0).

% is_class(TS, c0) :- has_proto(TS, p0, 1), has_proto(TS, p1, 0).
% is_class(TS, c1) :- has_proto(TS, p1, 1), has_proto(TS, p0, 0).
