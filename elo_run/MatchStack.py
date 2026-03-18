from sqlalchemy.orm import sessionmaker

from models import engine, Match

# Module-level cache: season -> list of match rows
_match_cache = {}


class MatchStack:
    """
    gets stack of matches in date order
    """

    def __init__(self, Season):
        self.engine = engine
        self.Season = Season

        if Season in _match_cache:
            self.matches = _match_cache[Season]
        else:
            Session = sessionmaker(bind=self.engine)
            session = Session()
            self.matches = list(
                session.query(
                    Match.id,
                    Match.mdid,
                    Match.stage,
                    Match.WTeamID,
                    Match.LTeamID,
                    Match.Delta,
                    Match.WLoc,
                    Match.WFGP_avg,
                    Match.WR_avg,
                    Match.WFGP3_avg,
                    Match.LR_avg,
                    Match.LFGP_avg,
                    Match.LFGP3_avg,
                    Match.WTO_margin_avg,
                    Match.LTO_margin_avg,
                    Match.WOR_avg,
                    Match.LOR_avg,
                    Match.WDR_avg,
                    Match.LDR_avg,
                )
                .filter(Match.Season == Season)
                # Only include regular season and tournament games
                .filter(Match.stage.in_(["T", "R"]))
                .order_by(Match.mdid)
                .all()
            )
            _match_cache[Season] = self.matches
            session.close()
