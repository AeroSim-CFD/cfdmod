from pydantic import BaseModel, ConfigDict


class WindProfile(BaseModel):
    """(Close to) abstract class for wind profile. Inherit to define implementation of get_U_H."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    U_H_overwrite: float | None = None

    @classmethod
    def build(cls, U_H_overwrite: float):
        return WindProfile(U_H_overwrite=U_H_overwrite)

    def get_U_H(
        self, height: float|None=None, direction: float|None=None, recurrence_period: float|None=None, use_kd: bool = True
    ) -> float:
        if self.U_H_overwrite is not None:
            return self.U_H_overwrite
        else:
            raise NotImplementedError("This class does not implement any calculation for U_H. Instantiate a child class or overwrite U_H")