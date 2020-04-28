# automatically generated by the FlatBuffers compiler, do not modify

# namespace: ppx

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Handshake(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsHandshake(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Handshake()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def HandshakeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x50\x50\x58\x46", size_prefixed=size_prefixed)

    # Handshake
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Handshake
    def SystemName(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def HandshakeStart(builder): builder.StartObject(1)
def HandshakeAddSystemName(builder, systemName): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(systemName), 0)
def HandshakeEnd(builder): return builder.EndObject()
