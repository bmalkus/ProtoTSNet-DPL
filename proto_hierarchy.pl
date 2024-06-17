nn(ptsnet, [TS, P])::has_proto(TS, P).

has_failure_A(TS) :- has_subfailure_AB(TS).
has_failure_A(TS) :- has_subfailure_AC(TS).
has_failure_A(TS) :- has_subfailure_AD(TS).

has_subfailure_AB(TS) :- has_proto(TS, p0).
has_subfailure_AC(TS) :- has_proto(TS, p1).
has_subfailure_AD(TS) :- has_proto(TS, p2), has_proto(TS, p3).

has_failure_B(TS) :- has_subfailure_BB(TS, p4).

has_subfailure_BB(TS, P) :- has_proto(TS, p5).

has_failure_C(TS) :- has_proto(TS, p6).
