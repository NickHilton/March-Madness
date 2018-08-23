UPDATE matches
SET Delta = WScore - LScore
where NumOT = 0;

UPDATE matches
SET Delta = 0
where NumOT > 0;
