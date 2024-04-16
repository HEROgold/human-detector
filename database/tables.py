import logging
import random
from typing import TYPE_CHECKING, List

import sqlalchemy
from sqlalchemy import (
    ForeignKey,
    Integer,
    String
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)

if TYPE_CHECKING:
    from camera import Camera as CameraObj

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
    room_name: Mapped[str] = mapped_column(String(255))
    cameras: Mapped[List["Camera"]] = relationship(back_populates="room")

    @property
    def human_count(self) -> int:
        """
        calculates this value when queried, on the python side. not visible in database
        
        Returns
        -------
        :class:`int`
            _description_
        """
        with Session(engine) as session:
            a = session.query(Camera.count).where(Camera.room_id == self.id).all()
            return sum([i[0] for i in a])

class Zone(Base):
    __tablename__ = "zone"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    zone_name: Mapped[str] = mapped_column(String(255))
    # cameras: Mapped[List["Camera"]] = relationship()


class Camera(Base):
    __tablename__ = "camera"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_name: Mapped[str] = mapped_column(String(255))
    count: Mapped[int] = mapped_column(Integer, default=0)
    room_id: Mapped[int] = mapped_column(ForeignKey(Room.id))
    # zones: Mapped[List["Zone"]] = relationship()

    room: Mapped["Room"] = relationship(back_populates="cameras")

    @classmethod
    def add_counter(cls, camera: "CameraObj") -> None:
        with Session(engine) as session:
            if cam := session.query(cls).where(cls.id == camera.camera_id).first():
                cam.count = camera.total_count
                session.commit()
                return
            session.add(cls(room_id=cls.id, count=camera.total_count))
            session.commit()


def test_database():
    room_names = [
        # Verdieping 1
        "Onderwijs ruimte 1",
        "Onderwijs ruimte 2",
        "Premium partners 1",
        "Server ruimte 1",
        "It 14",
        "Toiletten 1",
        "Toiletten 2",
        "Dakterras 1",
        "Dakterras 2",
        # Begane grond
        "IT 05",
        "Premium partners 2",
        "Kantine",
        "Keuken",
        "Presentatie ruimte",
        "Garderobe",
    ]
    zones = [
        "Hoofdingang",
        "Achteruitgang",
        "Toiletten",
        "Dakterras",
        "Kantine",
        "Keuken",
        "Presentatie ruimte",
        "Garderobe",
    ]
    cameras = [
        f"Camera {i}" for i in range(20)
    ]

    db_rooms = []
    db_zones = []
    db_cams = []

    with Session(engine) as session:
        for room in room_names:
            db_rooms.append(Room(room_name=room))
        for zone in zones:
            db_zones.append(Zone(zone_name=zone))

        session.add_all(db_rooms + db_zones)

        for cam in cameras:
            random_room_id = random.choice(session.query(Room).all()).id or 0
            db_cams.append(Camera(camera_name=cam, room_id=random_room_id, count=random.randint(0, 1000)))
        session.add_all(db_cams)
        session.commit()

        for i in session.query(Room).all():
            print(f"{i=}, {i.human_count=}")


Base().metadata.create_all(engine)


if __name__ == "__main__":
    test_database()
