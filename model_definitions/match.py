from sqlalchemy import Column, Integer, Float, VARCHAR, ForeignKey
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import relationship
from typing import List
from model_definitions import Base
from model_definitions.seed import Seed
from model_definitions.team import Team

stat_dict = {
    "id": "id",
    "Season": "Season",
    "mdid": "mdid",
    "stage": "stage",
    "WTeamID": "TeamID",
    "WScore": "Score",
    "WLoc": "Loc",
    "NumOT": "NumOT",
    "WFGP": "FGP",
    "WFGP_avg": "FGP_avg",
    "WFGP3": "FGP3",
    "WFGP3_avg": "FGP3_avg",
    "WR": "R",
    "WR_avg": "R_avg",
    "LTeamID": "TeamID",
    "LScore": "Score",
    "LFGP": "FGP",
    "LFGP_avg": "FGP_avg",
    "LFGP3": "FGP3",
    "LFGP3_avg": "FGP3_avg",
    "LR": "R",
    "LR_avg": "R_avg",
}

location_switch = {"H": "A", "N": "N", "A": "H"}


class Match(Base):
    """
    A given match with result and stats
    """
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    Season = Column(Integer, ForeignKey(Seed.Season))
    DayNum = Column(Integer)
    mdid = Column(Integer)
    stage = Column(VARCHAR(1))
    WTeamID = Column(Integer, ForeignKey(Team.TeamID))
    WScore = Column(Integer)
    LTeamID = Column(Integer, ForeignKey(Team.TeamID))
    LScore = Column(Integer)
    WLoc = Column(VARCHAR(1))
    NumOT = Column(Integer)
    WFGM = Column(Integer)
    WFGA = Column(Integer)
    WFGP = Column(Float)
    WFGP_avg = Column(Float)
    WFGM3 = Column(Integer)
    WFGA3 = Column(Integer)
    WFGP3 = Column(Float)
    WFGP3_avg = Column(Float)
    WFTM = Column(Integer)
    WFTA = Column(Integer)
    WOR = Column(Integer)
    WDR = Column(Integer)
    WR = Column(Integer)
    WR_avg = Column(Integer)
    WAst = Column(Integer)
    WTO = Column(Integer)
    WStl = Column(Integer)
    WBlk = Column(Integer)
    WPF = Column(Integer)
    LFGM = Column(Integer)
    LFGA = Column(Integer)
    LFGP = Column(Float)
    LFGP_avg = Column(Float)
    LFGM3 = Column(Integer)
    LFGA3 = Column(Integer)
    LFGP3 = Column(Float)
    LFGP3_avg = Column(Float)
    LFTM = Column(Integer)
    LFTA = Column(Integer)
    LOR = Column(Integer)
    LDR = Column(Integer)
    LR = Column(Float)
    LR_avg = Column(Float)
    LAst = Column(Integer)
    LTO = Column(Integer)
    LStl = Column(Integer)
    LBlk = Column(Integer)
    LPF = Column(Integer)
    Delta = Column(Integer)

    winning_team = relationship("Team", foreign_keys=[WTeamID])
    _winning_seed = relationship(
        "Seed",
        primaryjoin="and_(Match.WTeamID==Seed.TeamID, Match.Season==Seed.Season)",
        viewonly=True,
    )
    winning_seed = association_proxy("_winning_seed", "Seed")
    losing_team = relationship("Team", foreign_keys=[LTeamID])
    _losing_seed = relationship(
        "Seed",
        primaryjoin="and_(Match.LTeamID==Seed.TeamID, Match.Season==Seed.Season)",
        viewonly=True,
    )
    losing_seed = association_proxy("_losing_seed", "Seed")

    def teams(self) -> List[Team]:
        """
        Get teams involved in Match
        :return: list(Team, Team)
        """
        return [self.winning_team, self.losing_team]

    def _return_stats(self, cats: List[str]) -> dict:
        """
        Get stats from this match for a filtered set of categories
        -> Removes W/L from the stat

        :param cats: (list(str)) of stats to get
        :return: {stat: value}
        """
        stats = {}
        for cat in cats:
            stats[stat_dict[cat]] = self.__getattribute__(cat)
        return stats

    def winning_stats(self) -> dict:
        """
        Get all stats for the winning team
        :return: {stat: val}
        """
        cats = [
            "id",
            "Season",
            "mdid",
            "stage",
            "WTeamID",
            "WScore",
            "NumOT",
            "WFGP",
            "WFGP_avg",
            "WFGP3",
            "WFGP3_avg",
            "WR",
            "WR_avg",
        ]
        stats = self._return_stats(cats)
        stats["Loc"] = self.WLoc
        return stats

    def losing_stats(self):
        """
        Get all stats for the losing team

        :return: (dict) {stat: value}
        """
        cats = [
            "id",
            "Season",
            "mdid",
            "stage",
            "WTeamID",
            "WScore",
            "NumOT",
            "LFGP",
            "LFGP_avg",
            "LFGP3",
            "LFGP3_avg",
            "LR",
            "LR_avg",
        ]

        stats = self._return_stats(cats)
        stats["Loc"] = location_switch[self.WLoc]
        return stats
