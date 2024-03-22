import datetime
import logging

import sqlalchemy
from sqlalchemy import (
    Integer,
    DateTime
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
)


logger: logging.Logger = logging.getLogger("sqlalchemy.engine")


engine: sqlalchemy.Engine = sqlalchemy.create_engine("sqlite:///database/db.sqlite", echo=False)

# DB string refs
CASCADE = "CASCADE"


class Base(DeclarativeBase):
    "Subclass of DeclarativeBase with customizations."

    def __repr__(self) -> str:
        return str(self.__dict__)


class Room(Base):
    __tablename__ = "room"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    room_id: Mapped[int] = mapped_column(Integer)
    human_count: Mapped[int] = mapped_column(Integer, default=0)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime)


    @classmethod
    def add_counter(cls, room_id: int, count: int) -> None:
        with Session(engine) as session:
            session.add(cls(room_id=room_id, human_count=count, timestamp=datetime.datetime.now()))
            session.commit()

    def clear_older_than(self, days: int) -> None:
        with Session(engine) as session:
            session.query(Room).where(Room.timestamp < datetime.datetime.now() - datetime.timedelta(days=days)).delete()
            session.commit()

all_tables = Base.__subclasses__()

try:
    with Session(engine) as session:
        for i in all_tables:
            logger.debug(f"Checking for existing database table: {i}")
            session.query(i).first()
except Exception as e:
    logger.exception(f"Error getting all tables: {e}")
    """Should only run max once per startup, creating missing tables"""
    Base().metadata.create_all(engine)