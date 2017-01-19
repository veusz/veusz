#!/usr/bin/python
# -*- coding: utf-8 -*-

class OperationWrapper(object):
    """Helper class for operation-based objects like ToolPlugins or custom widgets"""
    name = 'OperationWrapper'
    _ops = False
    preserve = None


    @property
    def ops(self):
        if not self._ops:
            self._ops = []
        return self._ops

    @ops.setter
    def ops(self, val):
        self._ops = val

    def apply_ops(self, descr=False):
        from .. import document
        if not descr:
            if getattr(self, 'name', False):
                descr = self.name
            else:
                descr = self.typename
        if len(self.ops) > 0:
            self.doc.applyOperation(
                document.OperationMultiple(self.ops, descr=descr))
        self.ops = []

    def toset(self, out, name, val):
        """Set `name` to `val` on `out` widget"""
        from .. import document
        name = name.split('/')
        old = out.settings.getFromPath(name).get()
        if old != val:
            #            print 'preparing toset',name,old,val
            self.ops.append(document.OperationSettingSet(
                out.settings.getFromPath(name), val))
            return False
        return True

    def cpset(self, ref, out, name):
        """Copy setting `name` from `ref` to `out`"""
        val = ref.settings.getFromPath(name.split('/')).get()
        return self.toset(out, name, val)

    def eqset(self, ref, name):
        """Set DataPoint setting `name` equal to the same setting value on `ref` widget."""
        return self.cpset(ref, self, name)

    def dict_toset(self, out, props, preserve=None):
        if preserve is None:
            preserve = self.preserve
        pd = {}
        if preserve:
            pd = getattr(out, 'm_auto', {})
#         print 'found preserve dict',pd
        for k, v in pd.iteritems():
            if not props.has_key(k):
                continue
            cur = out.settings.getFromPath(k.split('/')).get()
            # It took a different value than the auto-arranged one
            if cur != v:
                #                 print 'dict_toset preserving',k,cur,v
                # Remove from auto assign
                props.pop(k)
        for name, value in props.iteritems():
            self.toset(out, name, value)
        # Update the m_auto attribute
        if len(pd) > 0:
            out.m_auto.update(props)
        # Create it if was missing
        elif preserve:
            out.m_auto = props
        return True


