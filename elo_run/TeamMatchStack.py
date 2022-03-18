from sqlalchemy.orm import sessionmaker

from models import engine, Team


class TeamMatchStack:
    def __init__(self, TeamID):

        self.engine = engine
        Session = sessionmaker(bind=self.engine)
        session = Session()

        self.TeamID = TeamID
        self.Team = session.query(Team).filter(Team.TeamID == TeamID).first()
        self.TeamName = self.Team.TeamName

        self.Season = 0
        self.FGP = 0
        self.FGP3 = 0
        self.season_matches = 0
        self.R = 0

        self.matches = list(self.Team.matches)

        session.close()

    def _season_averages(self, stats):
        """
        Get season averages up until the given game for the various stats
        :param stats: (dict)
        :return:
        """

        if self.Season != stats["Season"]:
            self.season_matches = 0

        if stats['FGP']:
            self.FGP = (self.FGP * self.season_matches + stats["FGP"]) / (
                self.season_matches + 1
            )
        if stats['FGP3']:
            self.FGP3 = (self.FGP3 * self.season_matches + stats["FGP3"]) / (
                self.season_matches + 1
            )
        if stats['R']:
            self.R = (self.R * self.season_matches + stats["R"]) / (self.season_matches + 1)
        self.Season = stats["Season"]
        self.season_matches = self.season_matches + 1

    def update_match_stats(self):

        Session = sessionmaker(bind=self.engine)
        session = Session()

        for match in self.Team.matches:
            if match.WTeamID == self.TeamID:
                stats = match.winning_stats()

                if stats["FGP"]:
                    match.WFGP3_avg = self.FGP3
                    match.WFGP_avg = self.FGP
                    match.WR_avg = self.R
                    session.merge(match)
                    self._season_averages(stats)

            else:
                stats = match.losing_stats()

                if stats["FGP"]:
                    match.LFGP3_avg = self.FGP3
                    match.LFGP_avg = self.FGP
                    match.LR_avg = self.R
                    session.merge(match)
                    self._season_averages(stats)

        session.commit()
        session.close()
