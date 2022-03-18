from sqlalchemy.orm import sessionmaker

from models import engine, Match


class MatchStack:
    """
    gets stack of matches in date order
    """

    def __init__(self, Season):
        self.engine = engine
        Session = sessionmaker(bind=self.engine)
        session = Session()
        self.Season = Season
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
            )
            .filter(Match.Season == Season)
            # Only include regular season and tournament games
            .filter(Match.stage.in_(["T", "R"]))
            .order_by(Match.mdid)
            .all()
        )

        session.close()
