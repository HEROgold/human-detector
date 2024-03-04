from datetime import datetime
import logging

import sqlalchemy
from sqlalchemy import (
    BigInteger,
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

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    human_count: Mapped[datetime] = mapped_column(DateTime)


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