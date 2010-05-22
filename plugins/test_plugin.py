from veusz.plugins.importplugin import ImportPlugin, importpluginregistry

class TestPlugin(ImportPlugin):
    name = 'test plugin'
    author = 'jeremy sanders'

    inputfields = ()
    checkoptions = ()

    dimensions = 1

importpluginregistry.append(TestPlugin)
