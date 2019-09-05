#
#
#

from ibllib.brainbox.core import Bunch

class IBLEphysChoiceWorldSession(Bunch):

    def __init__(self, *args, **kwargs):
        super(IBLEphysChoiceWorldSession, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        """Return a new Bunch instance which is a copy of the current Bunch instance."""
        return IBLEphysChoiceWorldSession(super(IBLEphysChoiceWorldSession, self).copy())
