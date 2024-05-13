nn(ptsnet,[TS, P],H,[0, 1])::has_proto(TS,P,H).
0.75::has_proto(ts0,p0).
0.65::has_proto(ts0,p1).
0.75::has_proto(ts1,p0).
0.35::has_proto(ts1,p1).
0.05::has_proto(ts2,p0).
0.8::has_proto(ts2,p1).
1.0::connected(p0,c0).
0.639602368528598::connected(p1,c1).
is_class(TS,c0) :- has_proto(TS,p0), connected(p0,c0).
is_class(TS,c1) :- has_proto(TS,p1), connected(p1,c1).
