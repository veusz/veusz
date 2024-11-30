/****************************************************************************
** 
** Copyright (c) 2009 Nokia Corporation and/or its subsidiary(-ies).
** All rights reserved.
** Contact: Nokia Corporation (qt-info@nokia.com)
** 
** This file is part of a Qt Solutions component.
**
** Commercial Usage  
** Licensees holding valid Qt Commercial licenses may use this file in
** accordance with the Qt Solutions Commercial License Agreement provided
** with the Software or, alternatively, in accordance with the terms
** contained in a written agreement between you and Nokia.
** 
** GNU Lesser General Public License Usage
** Alternatively, this file may be used under the terms of the GNU Lesser
** General Public License version 2.1 as published by the Free Software
** Foundation and appearing in the file LICENSE.LGPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU Lesser General Public License version 2.1 requirements
** will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
** 
** In addition, as a special exception, Nokia gives you certain
** additional rights. These rights are described in the Nokia Qt LGPL
** Exception version 1.1, included in the file LGPL_EXCEPTION.txt in this
** package.
** 
** GNU General Public License Usage 
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 3.0 as published by the Free Software
** Foundation and appearing in the file LICENSE.GPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU General Public License version 3.0 requirements will be
** met: http://www.gnu.org/copyleft/gpl.html.
** 
** Please note Third Party Software included with Qt Solutions may impose
** additional restrictions and it is the user's responsibility to ensure
** that they have met the licensing requirements of the GPL, LGPL, or Qt
** Solutions Commercial license and the relevant license of the Third
** Party Software they are using.
** 
** If you are unsure which license is appropriate for your use, please
** contact Nokia at qt-info@nokia.com.
** 
****************************************************************************/

#include <QtWidgets/QApplication>
#include <QtCore/QString>
#include <QtCore/QMap>
#include <QtGui/QScreen>
#include <QtGui/QPainter>
#include <QtGui/QPaintEvent>

#include "qtmmlwidget.h"

// *******************************************************************
// Declarations
// *******************************************************************

#define ROUND(a) (int)((a)+.5)

static bool           g_draw_frames			= false;
static const double   g_mfrac_spacing			= 0.1;
static const double   g_mroot_base_margin		= 0.1;
static const double   g_script_size_multiplier	    	= 0.7071; // sqrt(1/2)
static const int      g_min_font_point_size		= 8;
static const QChar    g_radical_char  	    	    	= QChar(0x1A, 0x22);
static const unsigned g_oper_spec_rows     	        = 9;

// use unnamed namespace
namespace
{

struct Mml
{
    enum NodeType {
	    NoNode = 0, MiNode, MnNode, MfracNode, MrowNode, MsqrtNode,
	    MrootNode, MsupNode, MsubNode, MsubsupNode, MoNode,
	    MstyleNode, TextNode, MphantomNode, MfencedNode,
	    MtableNode, MtrNode, MtdNode, MoverNode, MunderNode,
	    MunderoverNode, MerrorNode, MtextNode, MpaddedNode,
	    MspaceNode, MalignMarkNode, UnknownNode
    };

    enum MathVariant {
	    NormalMV	    = 0x0000,
	    BoldMV  	    = 0x0001,
	    ItalicMV	    = 0x0002,
	    DoubleStruckMV  = 0x0004,
	    ScriptMV        = 0x0008,
	    FrakturMV       = 0x0010,
	    SansSerifMV     = 0x0020,
	    MonospaceMV     = 0x0040
    };

    enum FormType { PrefixForm, InfixForm, PostfixForm };
    enum ColAlign { ColAlignLeft, ColAlignCenter, ColAlignRight };
    enum RowAlign { RowAlignTop, RowAlignCenter, RowAlignBottom,
		    RowAlignAxis, RowAlignBaseline };
    enum FrameType { FrameNone, FrameSolid, FrameDashed };

    struct FrameSpacing {
	FrameSpacing(int hor = 0, int ver = 0)
	    : m_hor(hor), m_ver(ver) {}
	int m_hor, m_ver;
    };
};

struct OperSpec {
    enum StretchDir { NoStretch, HStretch, VStretch, HVStretch };

    const char *name;
    Mml::FormType form;
    const char *attributes[g_oper_spec_rows];
    StretchDir stretch_dir;
};

struct NodeSpec
{
    Mml::NodeType type;
    const char *tag;
    const char *type_str;
    int child_spec;
    const char *child_types;
    const char *attributes;

    enum ChildSpec {
	    ChildAny     = -1, // any number of children allowed
	    ChildIgnore  = -2, // do not build subexpression of children
	    ImplicitMrow = -3  // if more than one child, build mrow
    };
};

struct EntitySpec
{
    const char *name;
    const char *value;
};

typedef QMap<QString, QString> MmlAttributeMap;
class MmlNode;
} // namespace

class MmlDocument : public Mml
{
    public:
	MmlDocument();
	~MmlDocument();
	void clear();

	bool setContent(QString text, QString *errorMsg = 0,
			    int *errorLine = 0, int *errorColumn = 0);
	void paint(QPainter *p, const QPoint &pos) const;
	void dump() const;
	QSize size() const;
	void layout();

	QString fontName(QtMmlWidget::MmlFont type) const;
	void setFontName(QtMmlWidget::MmlFont type, const QString &name);

	int baseFontPointSize() const
	    { return m_base_font_point_size; }
	void setBaseFontPointSize(int size)
	    { m_base_font_point_size = size; }
	QColor foregroundColor() const
	    { return m_foreground_color; }
	void setForegroundColor(const QColor &color)
	    { m_foreground_color = color; }
	QColor backgroundColor() const
	    { return m_background_color; }
	void setBackgroundColor(const QColor &color)
	    { m_background_color = color; }

    private:
	void _dump(const MmlNode *node, QString &indent) const;
	bool insertChild(MmlNode *parent, MmlNode *new_node, QString *errorMsg);

	MmlNode *domToMml(const QDomNode &dom_node, bool *ok, QString *errorMsg);
	MmlNode *createNode(NodeType type, const MmlAttributeMap &mml_attr,
				const QString &mml_value, QString *errorMsg);
	MmlNode *createImplicitMrowNode(const QDomNode &dom_node, bool *ok,
				    QString *errorMsg);

	void insertOperator(MmlNode *node, const QString &text);

	MmlNode *m_root_node;

	QString m_normal_font_name;
	QString m_fraktur_font_name;
	QString m_sans_serif_font_name;
	QString m_script_font_name;
	QString m_monospace_font_name;
	QString m_doublestruck_font_name;
	int m_base_font_point_size;
	QColor m_foreground_color;
	QColor m_background_color;
};

namespace {
class MmlNode : public Mml
{
    friend class ::MmlDocument;

    public:
	MmlNode(NodeType type, MmlDocument *document, const MmlAttributeMap &attribute_map);
	virtual ~MmlNode();

	// Mml stuff
	NodeType nodeType() const
	    { return m_node_type; }

	virtual QString toStr() const;

	void setRelOrigin(const QPoint &rel_origin);
	QPoint relOrigin() const
	    { return m_rel_origin; }
	void stretchTo(const QRect &rect);
	bool isStretched() const
	    { return m_stretched; }
	QPoint devicePoint(const QPoint &p) const;

	QRect myRect() const
	    { return m_my_rect; }
	QRect parentRect() const;
	virtual QRect deviceRect() const;
	void updateMyRect();
	virtual void setMyRect(const QRect &rect)
	    { m_my_rect = rect; }

	virtual void stretch();
	virtual void layout();
	virtual void paint(QPainter *p);

	int basePos() const;
	int overlinePos() const;
	int underlinePos() const;
	int em() const;
	int ex() const;

	QString explicitAttribute(const QString &name, const QString &def = QString()) const;
	QString inheritAttributeFromMrow(const QString &name, const QString &def = QString()) const;

	virtual QFont font() const;
	virtual QColor color() const;
	virtual QColor background() const;
	virtual int scriptlevel(const MmlNode *child = 0) const;


	// Node stuff
	MmlDocument *document() const
	    { return m_document; }
	MmlNode *parent() const
	    { return m_parent; }
	MmlNode *firstChild() const
	    { return m_first_child; }
	MmlNode *nextSibling() const
	    { return m_next_sibling; }
	MmlNode *previousSibling() const
	    { return m_previous_sibling; }
	MmlNode *lastSibling() const;
	MmlNode *firstSibling() const;
	bool isLastSibling() const
	    { return m_next_sibling == 0; }
	bool isFirstSibling() const
	    { return m_previous_sibling == 0; }
	bool hasChildNodes() const
	    { return m_first_child != 0; }

    protected:
	virtual void layoutSymbol();
	virtual void paintSymbol(QPainter *p) const;
	virtual QRect symbolRect() const
	    { return QRect(0, 0, 0, 0); }

	MmlNode *parentWithExplicitAttribute(const QString &name, NodeType type = NoNode);
	int interpretSpacing(const QString &value, bool *ok) const;

    private:
	MmlAttributeMap m_attribute_map;
	bool m_stretched;
	QRect m_my_rect, m_parent_rect;
	QPoint m_rel_origin;

	NodeType m_node_type;
	MmlDocument *m_document;

	MmlNode *m_parent,
		*m_first_child,
		*m_next_sibling,
		*m_previous_sibling;
};

class MmlTokenNode : public MmlNode
{
    public:
	MmlTokenNode(NodeType type, MmlDocument *document,
			const MmlAttributeMap &attribute_map)
	    : MmlNode(type, document, attribute_map) {}

	QString text() const;
};

class MmlMphantomNode : public MmlNode
{
    public:
	MmlMphantomNode(MmlDocument *document,
			    const MmlAttributeMap &attribute_map)
	    : MmlNode(MphantomNode, document, attribute_map) {}

	virtual void paint(QPainter *) {}
};

class MmlUnknownNode : public MmlNode
{
    public:
	MmlUnknownNode(MmlDocument *document,
			    const MmlAttributeMap &attribute_map)
	    : MmlNode(UnknownNode, document, attribute_map) {}
};

class MmlMfencedNode : public MmlNode
{
    public:
	MmlMfencedNode(MmlDocument *document,
			    const MmlAttributeMap &attribute_map)
	    : MmlNode(MfencedNode, document, attribute_map) {}
};

class MmlMalignMarkNode : public MmlNode
{
    public:
	MmlMalignMarkNode(MmlDocument *document)
	    : MmlNode(MalignMarkNode, document, MmlAttributeMap()) {}
};

class MmlMfracNode : public MmlNode
{
    public:
	MmlMfracNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
	    : MmlNode(MfracNode, document, attribute_map) {}

	MmlNode *numerator() const;
		MmlNode *denominator() const;

    protected:
	virtual void layoutSymbol();
	virtual void paintSymbol(QPainter *p) const;
	virtual QRect symbolRect() const;
};

class MmlMrowNode : public MmlNode
{
    public:
	MmlMrowNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		    : MmlNode(MrowNode, document, attribute_map) {}
};

class MmlRootBaseNode : public MmlNode
{
    public:
	MmlRootBaseNode(NodeType type, MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlNode(type, document, attribute_map) {}

	MmlNode *base() const;
	MmlNode *index() const;

	virtual int scriptlevel(const MmlNode *child = 0) const;

    protected:
	virtual void layoutSymbol();
	virtual void paintSymbol(QPainter *p) const;
	virtual QRect symbolRect() const;
	int tailWidth() const;
};

class MmlMrootNode : public MmlRootBaseNode
{
    public:
	MmlMrootNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlRootBaseNode(MrootNode, document, attribute_map) {}
};

class MmlMsqrtNode : public MmlRootBaseNode
{
    public:
	MmlMsqrtNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlRootBaseNode(MsqrtNode, document, attribute_map) {}

};


class MmlTextNode : public MmlNode
{
    public:
	MmlTextNode(const QString &text, MmlDocument *document);

	virtual QString toStr() const;
	QString text() const
	    { return m_text; }

	// TextNodes are not xml elements, so they can't have attributes of
	// their own. Everything is taken from the parent.
	virtual QFont font() const
	    { return parent()->font(); }
	virtual int scriptlevel(const MmlNode* = 0) const
	    { return parent()->scriptlevel(this); }
	virtual QColor color() const
	    { return parent()->color(); }
	virtual QColor background() const
	    { return parent()->background(); }

     protected:
	virtual void paintSymbol(QPainter *p) const;
	virtual QRect symbolRect() const;

	QString m_text;
};

class MmlMiNode : public MmlTokenNode
{
    public:
	MmlMiNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlTokenNode(MiNode, document, attribute_map) {}
};

class MmlMnNode : public MmlTokenNode
{
    public:
	MmlMnNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlTokenNode(MnNode, document, attribute_map) {}
};

class MmlSubsupBaseNode : public MmlNode
{
    public:
	MmlSubsupBaseNode(NodeType type, MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlNode(type, document, attribute_map) {}

	MmlNode *base() const;
	MmlNode *sscript() const;

	virtual int scriptlevel(const MmlNode *child = 0) const;
};

class MmlMsupNode : public MmlSubsupBaseNode
{
    public:
	MmlMsupNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlSubsupBaseNode(MsupNode, document, attribute_map) {}

    protected:
	virtual void layoutSymbol();
};

class MmlMsubNode : public MmlSubsupBaseNode
{
    public:
	MmlMsubNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlSubsupBaseNode(MsubNode, document, attribute_map) {}

    protected:
	virtual void layoutSymbol();
};

class MmlMsubsupNode : public MmlNode
{
    public:
	MmlMsubsupNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlNode(MsubsupNode, document, attribute_map) {}

	MmlNode *base() const;
	MmlNode *superscript() const;
	MmlNode *subscript() const;

	virtual int scriptlevel(const MmlNode *child = 0) const;

    protected:
	virtual void layoutSymbol();
};

class MmlMoNode : public MmlTokenNode
{
    public:
	MmlMoNode(MmlDocument *document, const MmlAttributeMap &attribute_map);

	QString dictionaryAttribute(const QString &name) const;
	virtual void stretch();
	virtual int lspace() const;
	virtual int rspace() const;

	virtual QString toStr() const;

    protected:
	virtual void layoutSymbol();
	virtual QRect symbolRect() const;

	virtual FormType form() const;

    private:
	const OperSpec *m_oper_spec;
};

class MmlMstyleNode : public MmlNode
{
    public:
	MmlMstyleNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlNode(MstyleNode, document, attribute_map) {}
};

class MmlTableBaseNode : public MmlNode
{
    public:
	MmlTableBaseNode(NodeType type, MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlNode(type, document, attribute_map) {}
};

class MmlMtableNode : public MmlTableBaseNode
{
    public:
	MmlMtableNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlTableBaseNode(MtableNode, document, attribute_map) {}

	int rowspacing() const;
	int columnspacing() const;
	int framespacing_hor() const;
	int framespacing_ver() const;
	FrameType frame() const;
	FrameType columnlines(int idx) const;
	FrameType rowlines(int idx) const;

    protected:
	virtual void layoutSymbol();
	virtual QRect symbolRect() const;
	virtual void paintSymbol(QPainter *p) const;

    private:
	struct CellSizeData
	{
	    void init(const MmlNode *first_row);
	    QList<int> col_widths, row_heights;
	    int numCols() const { return col_widths.count(); }
	    int numRows() const { return row_heights.count(); }
	    uint colWidthSum() const;
	    uint rowHeightSum() const;
	};

	CellSizeData m_cell_size_data;
	int m_content_width, m_content_height;
};

class MmlMtrNode : public MmlTableBaseNode
{
    public:
	MmlMtrNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlTableBaseNode(MtrNode, document, attribute_map) {}
    void layoutCells(const QList<int> &col_widths, int col_spc);
};

class MmlMtdNode : public MmlTableBaseNode
{
    public:
	MmlMtdNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlTableBaseNode(MtdNode, document, attribute_map)
	    { m_scriptlevel_adjust = 0; }
	virtual void setMyRect(const QRect &rect);

	ColAlign columnalign();
	RowAlign rowalign();
	uint colNum();
	uint rowNum();
	virtual int scriptlevel(const MmlNode *child = 0) const;

    private:
	int m_scriptlevel_adjust; // added or subtracted to scriptlevel to
				  // make contents fit the cell
};

class MmlMoverNode : public MmlNode
{
    public:
	MmlMoverNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlNode(MoverNode, document, attribute_map) {}
	virtual int scriptlevel(const MmlNode *node = 0) const;

    protected:
	virtual void layoutSymbol();
};

class MmlMunderNode : public MmlNode
{
    public:
	MmlMunderNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlNode(MunderNode, document, attribute_map) {}
	virtual int scriptlevel(const MmlNode *node = 0) const;

    protected:
	virtual void layoutSymbol();
};

class MmlMunderoverNode : public MmlNode
{
    public:
	MmlMunderoverNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlNode(MunderoverNode, document, attribute_map) {}
	virtual int scriptlevel(const MmlNode *node = 0) const;

    protected:
	virtual void layoutSymbol();
};

class MmlMerrorNode : public MmlNode
{
    public:
	MmlMerrorNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlNode(MerrorNode, document, attribute_map) {}
};

class MmlMtextNode : public MmlNode
{
    public:
	MmlMtextNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlNode(MtextNode, document, attribute_map) {}
};

class MmlMpaddedNode : public MmlNode
{
    public:
	MmlMpaddedNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlNode(MpaddedNode, document, attribute_map) {}

    public:
	int lspace() const;
	int width() const;
	int height() const;
	int depth() const;

    protected:
	int interpretSpacing(QString value, int base_value, bool *ok) const;
	virtual void layoutSymbol();
	virtual QRect symbolRect() const;
};

class MmlMspaceNode : public MmlNode
{
    public:
	MmlMspaceNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
		: MmlNode(MspaceNode, document, attribute_map) {}
};
}

static const NodeSpec *mmlFindNodeSpec(Mml::NodeType type);
static const NodeSpec *mmlFindNodeSpec(const QString &tag);
static bool mmlCheckChildType(Mml::NodeType parent_type,
			Mml::NodeType child_type, QString *error_str);
static bool mmlCheckAttributes(Mml::NodeType child_type,
			const MmlAttributeMap &attr, QString *error_str);
static QString mmlDictAttribute(const QString &name, const OperSpec *spec);
static const OperSpec *mmlFindOperSpec(const QString &name, Mml::FormType form);
static int interpretSpacing(QString name, int em, int ex, bool *ok);
static int interpretPercentSpacing(QString value, int base, bool *ok);
static uint interpretMathVariant(const QString &value, bool *ok);
static Mml::FormType interpretForm(const QString &value, bool *ok);
static Mml::FrameType interpretFrameType(const QString &value_list, uint idx, bool *ok);
static Mml::FrameSpacing interpretFrameSpacing(const QString &value_list, int em, int ex, bool *ok);
static Mml::ColAlign interpretColAlign(const QString &value_list, uint colnum, bool *ok);
static Mml::RowAlign interpretRowAlign(const QString &value_list, uint rownum, bool *ok);
static Mml::FrameType interpretFrameType(const QString &value_list, uint idx, bool *ok);
static QFont interpretDepreciatedFontAttr(const MmlAttributeMap &font_attr, QFont &fn, int em, int ex);
static QFont interpretMathSize(QString value, QFont &fn, int em, int ex, bool *ok);
static QString interpretListAttr(const QString &value_list, int idx, const QString &def);
static QString rectToStr(const QRect &rect);
static QString entityDeclarations();


#define MML_ATT_COMMON      " class style id xref actiontype "
#define MML_ATT_FONTSIZE    " fontsize fontweight fontstyle fontfamily color "
#define MML_ATT_MATHVARIANT " mathvariant mathsize mathcolor mathbackground "
#define MML_ATT_FONTINFO    MML_ATT_FONTSIZE MML_ATT_MATHVARIANT
#define MML_ATT_OPINFO      " form fence separator lspace rspace stretchy symmetric " \
			    " maxsize minsize largeop movablelimits accent "
#define MML_ATT_SIZEINFO    " width height depth "
#define MML_ATT_TABLEINFO   " align rowalign columnalign columnwidth groupalign " \
			    " alignmentscope side rowspacing columnspacing rowlines " \
			    " columnlines width frame framespacing equalrows " \
			    " equalcolumns displaystyle "
#define MML_ATT_MFRAC       " bevelled numalign denomalign linethickness "
#define MML_ATT_MSTYLE	    MML_ATT_FONTINFO MML_ATT_OPINFO \
			    " scriptlevel lquote rquote linethickness displaystyle " \
			    " scriptsizemultiplier scriptminsize background " \
			    " veryverythinmathspace verythinmathspace thinmathspace " \
			    " mediummathspace thickmathspace verythickmathspace " \
			    " veryverythickmathspace open close separators " \
			    " subscriptshift superscriptshift accentunder tableinfo " \
			    " rowspan columnspan edge selection bevelled "
#define MML_ATT_MTABLE	    " align rowalign columnalign groupalign alignmentscope " \
			    " columnwidth width rowspacing columnspacing rowlines columnlines " \
			    " frame framespacing equalrows equalcolumns displaystyle side " \
			    " minlabelspacing "

static const NodeSpec g_node_spec_data[] = {

//  	type	                tag             type_str            child_spec              child_types              attributes ""=none, 0=any
//      ----------------------- --------------- ------------------- ----------------------- ------------------------ ------------------------------------
    {	Mml::MiNode,	    	"mi",	    	"MiNode",   	    NodeSpec::ChildAny,      " TextNode MalignMark ", MML_ATT_COMMON MML_ATT_FONTINFO     },
    {	Mml::MnNode,	    	"mn",	    	"MnNode",   	    NodeSpec::ChildAny,      " TextNode MalignMark ", MML_ATT_COMMON MML_ATT_FONTINFO     },
    {	Mml::MfracNode,   	"mfrac",    	"MfracNode",	    2,   		    0,                       MML_ATT_COMMON MML_ATT_MFRAC         },
    {	Mml::MrowNode,    	"mrow",     	"MrowNode", 	    NodeSpec::ChildAny,     0,                       MML_ATT_COMMON " display mode "      },
    {	Mml::MsqrtNode,   	"msqrt",    	"MsqrtNode",	    NodeSpec::ImplicitMrow, 0,                       MML_ATT_COMMON                      },
    {	Mml::MrootNode,   	"mroot",    	"MrootNode",	    2,   		    0,                       MML_ATT_COMMON                       },
    {	Mml::MsupNode,    	"msup",     	"MsupNode", 	    2,	    	    	    0,                       MML_ATT_COMMON " subscriptshift "    },
    {	Mml::MsubNode,    	"msub",     	"MsubNode", 	    2,	    	    	    0,                       MML_ATT_COMMON " superscriptshift "  },
    {	Mml::MsubsupNode, 	"msubsup",  	"MsubsupNode",	    3,	    	    	    0,                       MML_ATT_COMMON " subscriptshift superscriptshift " },
    {	Mml::MoNode,	    	"mo",	    	"MoNode",   	    NodeSpec::ChildAny,     " TextNode MalignMark ", MML_ATT_COMMON MML_ATT_FONTINFO MML_ATT_OPINFO    },
    {	Mml::MstyleNode,  	"mstyle",   	"MstyleNode",	    NodeSpec::ImplicitMrow, 0,                       MML_ATT_COMMON MML_ATT_MSTYLE       },
    {	Mml::MphantomNode,  	"mphantom",   	"MphantomNode",	    NodeSpec::ImplicitMrow, 0,                       MML_ATT_COMMON         	    	  },
    {	Mml::MalignMarkNode,	"malignmark",	"MalignMarkNode",   0,	    	    	    0,	    	             ""	    	    	    	         },
    {	Mml::MfencedNode,	"mfenced",	"MfencedNode",      NodeSpec::ChildAny,	    0,	    	             MML_ATT_COMMON " open close separators "	        },
    {	Mml::MtableNode,  	"mtable",   	"MtableNode",	    NodeSpec::ChildAny,	    " MtrNode ",    	     MML_ATT_COMMON MML_ATT_MTABLE  	 },
    {	Mml::MtrNode,  	        "mtr",   	"MtrNode",	    NodeSpec::ChildAny,	    " MtdNode ",    	     MML_ATT_COMMON " rowalign columnalign groupalign " },
    {	Mml::MtdNode,  	        "mtd",   	"MtdNode",	    NodeSpec::ImplicitMrow, 0,    	     	     MML_ATT_COMMON " rowspan columnspan rowalign columnalign groupalign " },
    {	Mml::MoverNode,  	"mover",   	"MoverNode",	    2,	    	    	    0,    	     	     MML_ATT_COMMON " accent "	    	 },
    {	Mml::MunderNode,  	"munder",   	"MunderNode",	    2,	    	    	    0,    	     	     MML_ATT_COMMON " accentunder "	 },
    {	Mml::MunderoverNode,    "munderover",   "MunderoverNode",   3,	    	    	    0,    	     	     MML_ATT_COMMON " accentunder accent " },
    {	Mml::MerrorNode,  	"merror",   	"MerrorNode",	    NodeSpec::ImplicitMrow, 0,                       MML_ATT_COMMON 	    	    	 },
    {	Mml::MtextNode,  	"mtext",   	"MtextNode",	    1,	    	    	    " TextNode ",            MML_ATT_COMMON " width height depth linebreak " },
    {	Mml::MpaddedNode,  	"mpadded",   	"MpaddedNode",	    NodeSpec::ImplicitMrow, 0,                       MML_ATT_COMMON " width height depth lspace " },
    {	Mml::MspaceNode,  	"mspace",   	"MspaceNode",	    NodeSpec::ImplicitMrow, 0,                       MML_ATT_COMMON " width height depth linebreak " },
    {	Mml::TextNode,    	0,  	    	"TextNode", 	    NodeSpec::ChildIgnore,  0,                       ""                                  },
    {	Mml::UnknownNode,	0,	      	"UnknownNode",      NodeSpec::ChildAny,     0,                       0         	    	    	         },
    {	Mml::NoNode,	    	0,	      	0,  	    	    0,       	            0,                       0         	    	    	         }
};

static const char *g_oper_spec_names[g_oper_spec_rows] = {                          	     "accent",                 "fence",               "largeop",                "lspace",               "minsize",         "movablelimits",                "rspace",             "separator",              "stretchy"	    /* stretchdir */ };
static const OperSpec g_oper_spec_data[] = {

	{ "!!"		    	    	,		Mml::PostfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "!!"
	{ "!"		    	    	,		Mml::PostfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "!"
	{ "!="		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "!="
	{ "&And;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,       "mediummathspace",			0,			 0,	  "mediummathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&And;"
	{ "&ApplyFunction;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "&ApplyFunction;"
	{ "&Assign;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Assign;"
	{ "&Backslash;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&Backslash;"
	{ "&Because;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Because;"
	{ "&Breve;"		    	,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "&Breve;"
	{ "&Cap;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Cap;"
	{ "&CapitalDifferentialD;"       ,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // &CapitalDifferentialD;"
	{ "&Cedilla;"		    	,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "&Cedilla;"
	{ "&CenterDot;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&CenterDot;"
	{ "&CircleDot;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&CircleDot;"
	{ "&CircleMinus;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&CircleMinus;"
	{ "&CirclePlus;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&CirclePlus;"
	{ "&CircleTimes;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&CircleTimes;"
	{ "&ClockwiseContourIntegral;"   ,		 Mml::PrefixForm, {			    0,  		     0, 		 "true",		   "0em",			0,			 0,		      "0em",			   0,			"true"},            OperSpec::VStretch }, // &ClockwiseContourIntegral;"
	{ "&CloseCurlyDoubleQuote;"      ,		Mml::PostfixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // &CloseCurlyDoubleQuote;"
	{ "&CloseCurlyQuote;"		,		Mml::PostfixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "&CloseCurlyQuote;"
	{ "&Colon;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Colon;"
	{ "&Congruent;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Congruent;"
	{ "&ContourIntegral;"		,		 Mml::PrefixForm, {			    0,  		     0, 		 "true",		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "&ContourIntegral;"
	{ "&Coproduct;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Coproduct;"
	{ "&CounterClockwiseContourIntegral;",		 Mml::PrefixForm, {		            0,  		     0, 		 "true",		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // &CounterClockwiseContourInteg
	{ "&Cross;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Cross;"
	{ "&Cup;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Cup;"
	{ "&CupCap;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&CupCap;"
	{ "&Del;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Del;"
	{ "&DiacriticalAcute;"		,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "&DiacriticalAcute;"
	{ "&DiacriticalDot;"		,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "&DiacriticalDot;"
	{ "&DiacriticalDoubleAcute;" 	,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // &DiacriticalDoubleAcute;"
	{ "&DiacriticalGrave;"		,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "&DiacriticalGrave;"
	{ "&DiacriticalLeftArrow;"   	,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // &DiacriticalLeftArrow;"
	{ "&DiacriticalLeftRightArrow;"  ,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // &DiacriticalLeftRightArrow;"
	{ "&DiacriticalLeftRightVector;" ,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // &DiacriticalLeftRightVector;"
	{ "&DiacriticalLeftVector;"      ,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // &DiacriticalLeftVector;"
	{ "&DiacriticalRightArrow;"  	,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // &DiacriticalRightArrow;"
	{ "&DiacriticalRightVector;" 	,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // &DiacriticalRightVector;"
	{ "&DiacriticalTilde;"		,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },           OperSpec::NoStretch }, // "&DiacriticalTilde;"
	{ "&Diamond;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Diamond;"
	{ "&DifferentialD;"		,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&DifferentialD;"
	{ "&DotEqual;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&DotEqual;"
	{ "&DoubleContourIntegral;"      ,		 Mml::PrefixForm, {			    0,  		     0, 		 "true",		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // &DoubleContourIntegral;"
	{ "&DoubleDot;"		    	,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "&DoubleDot;"
	{ "&DoubleDownArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&DoubleDownArrow;"
	{ "&DoubleLeftArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&DoubleLeftArrow;"
	{ "&DoubleLeftRightArrow;"       ,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // &DoubleLeftRightArrow;"
	{ "&DoubleLeftTee;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&DoubleLeftTee;"
	{ "&DoubleLongLeftArrow;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&DoubleLongLeftArrow;"
	{ "&DoubleLongLeftRightArrow;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::HStretch }, // &DoubleLongLeftRightArrow;"
	{ "&DoubleLongRightArrow;"       ,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::HStretch }, // &DoubleLongRightArrow;"
	{ "&DoubleRightArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&DoubleRightArrow;"
	{ "&DoubleRightTee;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&DoubleRightTee;"
	{ "&DoubleUpArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&DoubleUpArrow;"
	{ "&DoubleUpDownArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&DoubleUpDownArrow;"
	{ "&DoubleVerticalBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&DoubleVerticalBar;"
	{ "&DownArrow;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&DownArrow;"
	{ "&DownArrowBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&DownArrowBar;"
	{ "&DownArrowUpArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&DownArrowUpArrow;"
	{ "&DownBreve;"		    	,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "&DownBreve;"
	{ "&DownLeftRightVector;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&DownLeftRightVector;"
	{ "&DownLeftTeeVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&DownLeftTeeVector;"
	{ "&DownLeftVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&DownLeftVector;"
	{ "&DownLeftVectorBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&DownLeftVectorBar;"
	{ "&DownRightTeeVector;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&DownRightTeeVector;"
	{ "&DownRightVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&DownRightVector;"
	{ "&DownRightVectorBar;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&DownRightVectorBar;"
	{ "&DownTee;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&DownTee;"
	{ "&DownTeeArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&DownTeeArrow;"
	{ "&Element;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Element;"
	{ "&Equal;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Equal;"
	{ "&EqualTilde;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&EqualTilde;"
	{ "&Equilibrium;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&Equilibrium;"
	{ "&Exists;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Exists;"
	{ "&ForAll;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&ForAll;"
	{ "&GreaterEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&GreaterEqual;"
	{ "&GreaterEqualLess;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&GreaterEqualLess;"
	{ "&GreaterFullEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&GreaterFullEqual;"
	{ "&GreaterGreater;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&GreaterGreater;"
	{ "&GreaterLess;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&GreaterLess;"
	{ "&GreaterSlantEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&GreaterSlantEqual;"
	{ "&GreaterTilde;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&GreaterTilde;"
	{ "&Hacek;"		    	,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },           OperSpec::NoStretch }, // "&Hacek;"
	{ "&Hat;"		    	,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // "&Hat;"
	{ "&HorizontalLine;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,		   "0em",		      "0",			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // "&HorizontalLine;"
	{ "&HumpDownHump;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&HumpDownHump;"
	{ "&HumpEqual;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&HumpEqual;"
	{ "&Implies;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&Implies;"
	{ "&Integral;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		 "true",		   "0em",			0,			 0,		      "0em",			   0,		            0 },           OperSpec::NoStretch }, // "&Integral;"
	{ "&Intersection;"		,		 Mml::PrefixForm, {			    0,  		     0, 		 "true",		   "0em",			0,		    "true",	    "thinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&Intersection;"
	{ "&InvisibleComma;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,		      "0em",		      "true",			    0 },           OperSpec::NoStretch }, // "&InvisibleComma;"
	{ "&InvisibleTimes;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "&InvisibleTimes;"
	{ "&LeftAngleBracket;"		,		 Mml::PrefixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "&LeftAngleBracket;"
	{ "&LeftArrow;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&LeftArrow;"
	{ "&LeftArrowBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&LeftArrowBar;"
	{ "&LeftArrowRightArrow;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&LeftArrowRightArrow;"
	{ "&LeftBracketingBar;"		,		 Mml::PrefixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "&LeftBracketingBar;"
	{ "&LeftCeiling;"		,		 Mml::PrefixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "&LeftCeiling;"
	{ "&LeftDoubleBracket;"		,		 Mml::PrefixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // "&LeftDoubleBracket;"
	{ "&LeftDoubleBracketingBar;"    ,		 Mml::PrefixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // &LeftDoubleBracketingBar;"
	{ "&LeftDownTeeVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&LeftDownTeeVector;"
	{ "&LeftDownVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&LeftDownVector;"
	{ "&LeftDownVectorBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&LeftDownVectorBar;"
	{ "&LeftFloor;"		    	,		 Mml::PrefixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "&LeftFloor;"
	{ "&LeftRightArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&LeftRightArrow;"
	{ "&LeftRightVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&LeftRightVector;"
	{ "&LeftSkeleton;"		,		 Mml::PrefixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "&LeftSkeleton;"
	{ "&LeftTee;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&LeftTee;"
	{ "&LeftTeeArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&LeftTeeArrow;"
	{ "&LeftTeeVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&LeftTeeVector;"
	{ "&LeftTriangle;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&LeftTriangle;"
	{ "&LeftTriangleBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&LeftTriangleBar;"
	{ "&LeftTriangleEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&LeftTriangleEqual;"
	{ "&LeftUpDownVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&LeftUpDownVector;"
	{ "&LeftUpTeeVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&LeftUpTeeVector;"
	{ "&LeftUpVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&LeftUpVector;"
	{ "&LeftUpVectorBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&LeftUpVectorBar;"
	{ "&LeftVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&LeftVector;"
	{ "&LeftVectorBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&LeftVectorBar;"
	{ "&LessEqualGreater;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&LessEqualGreater;"
	{ "&LessFullEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&LessFullEqual;"
	{ "&LessGreater;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&LessGreater;"
	{ "&LessLess;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&LessLess;"
	{ "&LessSlantEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&LessSlantEqual;"
	{ "&LessTilde;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&LessTilde;"
	{ "&LongLeftArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&LongLeftArrow;"
	{ "&LongLeftRightArrow;"    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&LongLeftRightArrow;"
	{ "&LongRightArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&LongRightArrow;"
	{ "&LowerLeftArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&LowerLeftArrow;"
	{ "&LowerRightArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&LowerRightArrow;"
	{ "&MinusPlus;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0, "veryverythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&MinusPlus;"
	{ "&NestedGreaterGreater;"       ,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // &NestedGreaterGreater;"
	{ "&NestedLessLess;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NestedLessLess;"
	{ "&Not;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Not;"
	{ "&NotCongruent;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotCongruent;"
	{ "&NotCupCap;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotCupCap;"
	{ "&NotDoubleVerticalBar;"       ,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // &NotDoubleVerticalBar;"
	{ "&NotElement;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotElement;"
	{ "&NotEqual;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotEqual;"
	{ "&NotEqualTilde;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotEqualTilde;"
	{ "&NotExists;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotExists;"
	{ "&NotGreater;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotGreater;"
	{ "&NotGreaterEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotGreaterEqual;"
	{ "&NotGreaterFullEqual;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotGreaterFullEqual;"
	{ "&NotGreaterGreater;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotGreaterGreater;"
	{ "&NotGreaterLess;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotGreaterLess;"
	{ "&NotGreaterSlantEqual;"       ,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // &NotGreaterSlantEqual;"
	{ "&NotGreaterTilde;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotGreaterTilde;"
	{ "&NotHumpDownHump;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotHumpDownHump;"
	{ "&NotHumpEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotHumpEqual;"
	{ "&NotLeftTriangle;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotLeftTriangle;"
	{ "&NotLeftTriangleBar;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotLeftTriangleBar;"
	{ "&NotLeftTriangleEqual;"       ,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // &NotLeftTriangleEqual;"
	{ "&NotLess;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotLess;"
	{ "&NotLessEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotLessEqual;"
	{ "&NotLessFullEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotLessFullEqual;"
	{ "&NotLessGreater;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotLessGreater;"
	{ "&NotLessLess;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotLessLess;"
	{ "&NotLessSlantEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotLessSlantEqual;"
	{ "&NotLessTilde;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotLessTilde;"
	{ "&NotNestedGreaterGreater;"    ,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // &NotNestedGreaterGreater;"
	{ "&NotNestedLessLess;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotNestedLessLess;"
	{ "&NotPrecedes;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotPrecedes;"
	{ "&NotPrecedesEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotPrecedesEqual;"
	{ "&NotPrecedesSlantEqual;"      ,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // &NotPrecedesSlantEqual;"
	{ "&NotPrecedesTilde;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotPrecedesTilde;"
	{ "&NotReverseElement;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotReverseElement;"
	{ "&NotRightTriangle;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotRightTriangle;"
	{ "&NotRightTriangleBar;"   	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotRightTriangleBar;"
	{ "&NotRightTriangleEqual;"  	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // &NotRightTriangleEqual;"
	{ "&NotSquareSubset;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotSquareSubset;"
	{ "&NotSquareSubsetEqual;"   	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // &NotSquareSubsetEqual;"
	{ "&NotSquareSuperset;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotSquareSuperset;"
	{ "&NotSquareSupersetEqual;"     ,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // &NotSquareSupersetEqual;"
	{ "&NotSubset;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotSubset;"
	{ "&NotSubsetEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotSubsetEqual;"
	{ "&NotSucceeds;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotSucceeds;"
	{ "&NotSucceedsEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotSucceedsEqual;"
	{ "&NotSucceedsSlantEqual;"      ,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // &NotSucceedsSlantEqual;"
	{ "&NotSucceedsTilde;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotSucceedsTilde;"
	{ "&NotSuperset;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotSuperset;"
	{ "&NotSupersetEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotSupersetEqual;"
	{ "&NotTilde;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotTilde;"
	{ "&NotTildeEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotTildeEqual;"
	{ "&NotTildeFullEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotTildeFullEqual;"
	{ "&NotTildeTilde;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotTildeTilde;"
	{ "&NotVerticalBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&NotVerticalBar;"
	{ "&OpenCurlyDoubleQuote;"       ,		 Mml::PrefixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // &OpenCurlyDoubleQuote;"
	{ "&OpenCurlyQuote;"		,		 Mml::PrefixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "&OpenCurlyQuote;"
	{ "&Or;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,       "mediummathspace",			0,			 0,	  "mediummathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&Or;"
	{ "&OverBar;"		    	,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // "&OverBar;"
	{ "&OverBrace;"		    	,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // "&OverBrace;"
	{ "&OverBracket;"		,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // "&OverBracket;"
	{ "&OverParenthesis;"		,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "&OverParenthesis;"
	{ "&PartialD;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&PartialD;"
	{ "&PlusMinus;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0, "veryverythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&PlusMinus;"
	{ "&Precedes;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Precedes;"
	{ "&PrecedesEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&PrecedesEqual;"
	{ "&PrecedesSlantEqual;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&PrecedesSlantEqual;"
	{ "&PrecedesTilde;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&PrecedesTilde;"
	{ "&Product;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		 "true",		   "0em",			0,		    "true",	    "thinmathspace",			   0,		            0 },           OperSpec::NoStretch }, // "&Product;"
	{ "&Proportion;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Proportion;"
	{ "&Proportional;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Proportional;"
	{ "&ReverseElement;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&ReverseElement;"
	{ "&ReverseEquilibrium;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&ReverseEquilibrium;"
	{ "&ReverseUpEquilibrium;"   	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::HStretch }, // &ReverseUpEquilibrium;"
	{ "&RightAngleBracket;"		,		Mml::PostfixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "&RightAngleBracket;"
	{ "&RightArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&RightArrow;"
	{ "&RightArrowBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&RightArrowBar;"
	{ "&RightArrowLeftArrow;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&RightArrowLeftArrow;"
	{ "&RightBracketingBar;"	,		Mml::PostfixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "&RightBracketingBar;"
	{ "&RightCeiling;"		,		Mml::PostfixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "&RightCeiling;"
	{ "&RightDoubleBracket;"	,		Mml::PostfixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "&RightDoubleBracket;"
	{ "&RightDoubleBracketingBar;"   ,		Mml::PostfixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // &RightDoubleBracketingBar;"
	{ "&RightDownTeeVector;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&RightDownTeeVector;"
	{ "&RightDownVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&RightDownVector;"
	{ "&RightDownVectorBar;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&RightDownVectorBar;"
	{ "&RightFloor;"		,		Mml::PostfixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "&RightFloor;"
	{ "&RightSkeleton;"		,		Mml::PostfixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "&RightSkeleton;"
	{ "&RightTee;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&RightTee;"
	{ "&RightTeeArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&RightTeeArrow;"
	{ "&RightTeeVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&RightTeeVector;"
	{ "&RightTriangle;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&RightTriangle;"
	{ "&RightTriangleBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&RightTriangleBar;"
	{ "&RightTriangleEqual;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&RightTriangleEqual;"
	{ "&RightUpDownVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&RightUpDownVector;"
	{ "&RightUpTeeVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&RightUpTeeVector;"
	{ "&RightUpVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&RightUpVector;"
	{ "&RightUpVectorBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&RightUpVectorBar;"
	{ "&RightVector;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&RightVector;"
	{ "&RightVectorBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::HStretch }, // "&RightVectorBar;"
	{ "&RoundImplies;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&RoundImplies;"
	{ "&ShortDownArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&ShortDownArrow;"
	{ "&ShortLeftArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },            OperSpec::HStretch }, // "&ShortLeftArrow;"
	{ "&ShortRightArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },            OperSpec::HStretch }, // "&ShortRightArrow;"
	{ "&ShortUpArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,			    0 },            OperSpec::VStretch }, // "&ShortUpArrow;"
	{ "&SmallCircle;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&SmallCircle;"
	{ "&Sqrt;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&Sqrt;"
	{ "&Square;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Square;"
	{ "&SquareIntersection;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,       "mediummathspace",			0,			 0,	  "mediummathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&SquareIntersection;"
	{ "&SquareSubset;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&SquareSubset;"
	{ "&SquareSubsetEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&SquareSubsetEqual;"
	{ "&SquareSuperset;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&SquareSuperset;"
	{ "&SquareSupersetEqual;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&SquareSupersetEqual;"
	{ "&SquareUnion;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,       "mediummathspace",			0,			 0,	  "mediummathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&SquareUnion;"
	{ "&Star;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Star;"
	{ "&Subset;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Subset;"
	{ "&SubsetEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&SubsetEqual;"
	{ "&Succeeds;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Succeeds;"
	{ "&SucceedsEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&SucceedsEqual;"
	{ "&SucceedsSlantEqual;"	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&SucceedsSlantEqual;"
	{ "&SucceedsTilde;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&SucceedsTilde;"
	{ "&SuchThat;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&SuchThat;"
	{ "&Sum;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		 "true",		   "0em",			0,		    "true",	    "thinmathspace",			   0,		            0 },           OperSpec::NoStretch }, // "&Sum;"
	{ "&Superset;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Superset;"
	{ "&SupersetEqual;"		,   	    	  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&SupersetEqual;"
	{ "&Therefore;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Therefore;"
	{ "&Tilde;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Tilde;"
	{ "&TildeEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&TildeEqual;"
	{ "&TildeFullEqual;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&TildeFullEqual;"
	{ "&TildeTilde;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&TildeTilde;"
	{ "&TripleDot;"		    	,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "&TripleDot;"
	{ "&UnderBar;"		    	,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // "&UnderBar;"
	{ "&UnderBrace;"		,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // "&UnderBrace;"
	{ "&UnderBracket;"		,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },           OperSpec::HStretch }, // "&UnderBracket;"
	{ "&UnderParenthesis;"		,		Mml::PostfixForm, {		       "true",  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::HStretch }, // "&UnderParenthesis;"
	{ "&Union;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		 "true",		   "0em",			0,		    "true",	    "thinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&Union;"
	{ "&UnionPlus;"		    	,		 Mml::PrefixForm, {			    0,  		     0, 		 "true",		   "0em",			0,		    "true",	    "thinmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&UnionPlus;"
	{ "&UpArrow;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&UpArrow;"
	{ "&UpArrowBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&UpArrowBar;"
	{ "&UpArrowDownArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&UpArrowDownArrow;"
	{ "&UpDownArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&UpDownArrow;"
	{ "&UpEquilibrium;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&UpEquilibrium;"
	{ "&UpTee;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&UpTee;"
	{ "&UpTeeArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&UpTeeArrow;"
	{ "&UpperLeftArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&UpperLeftArrow;"
	{ "&UpperRightArrow;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },           OperSpec::HVStretch }, // "&UpperRightArrow;"
	{ "&Vee;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Vee;"
	{ "&VerticalBar;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&VerticalBar;"
	{ "&VerticalLine;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,		   "0em",		      "0",			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "&VerticalLine;"
	{ "&VerticalSeparator;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "&VerticalSeparator;"
	{ "&VerticalTilde;"		,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&VerticalTilde;"
	{ "&Wedge;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&Wedge;"
	{ "&amp;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&amp;"
	{ "&amp;&amp;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&amp;&amp;"
	{ "&le;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&le;"
	{ "&lt;"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&lt;"
	{ "&lt;="		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&lt;="
	{ "&lt;>"		    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "&lt;>"
	{ "'"		    	    	,		Mml::PostfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "'"
	{ "("		    	    	,		 Mml::PrefixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "("
	{ ")"		    	    	,		Mml::PostfixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // ")"
	{ "*"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "*"
	{ "**"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "**"
	{ "*="		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "*="
	{ "+"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "+"
	{ "+"		    	    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0, "veryverythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "+"
	{ "++"		    	    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "++"
	{ "+="		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "+="
	{ ","		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,    "verythickmathspace",		      "true",			    0 },           OperSpec::NoStretch }, // ","
	{ "-"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "-"
	{ "-"		    	    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0, "veryverythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "-"
	{ "--"		    	    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, //   "--"
	{ "-="		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "-="
	{ "->"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "->"
	{ "."		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "."
	{ ".."		    	    	,		Mml::PostfixForm, {			    0,  		     0, 		      0,       "mediummathspace",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // ".."
	{ "..."		    	    	,		Mml::PostfixForm, {			    0,  		     0, 		      0,       "mediummathspace",			0,			 0,		      "0em",			   0,			    0 },           OperSpec::NoStretch }, // "..."
	{ "/"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	 "thinmathspace",			0,			 0,	    "thinmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "/"
	{ "//"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "//"
	{ "/="		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "/="
	{ ":"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // ":"
	{ ":="		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // ":="
	{ ";"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,    "verythickmathspace",		      "true",			    0 },           OperSpec::NoStretch }, // ";"
	{ ";"		    	    	,		Mml::PostfixForm, {			    0,  		     0, 		      0,		   "0em",			0,			 0,		      "0em",		      "true",			    0 },           OperSpec::NoStretch }, // ";"
	{ "="		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "="
	{ "=="		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "=="
	{ ">"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // ">"
	{ ">="		                ,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,			    0 },           OperSpec::NoStretch }, // ">="
	{ "?"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "?"
	{ "@"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "@"
	{ "["		    	    	,		 Mml::PrefixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "["
	{ "]"		    	    	,		Mml::PostfixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "]"
	{ "^"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "^"
	{ "_"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "_"
	{ "lim"		    	    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,		    "true",	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "lim"
	{ "max"		    	    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,		    "true",	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "max"
	{ "min"		    	    	,		 Mml::PrefixForm, {			    0,  		     0, 		      0,		   "0em",			0,		    "true",	    "thinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "min"
	{ "{"              		,		 Mml::PrefixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "{"
	{ "|"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,	"thickmathspace",			0,			 0,	   "thickmathspace",			   0,		       "true" },            OperSpec::VStretch }, // "|"
	{ "||"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,       "mediummathspace",			0,			 0,	  "mediummathspace",			   0,			    0 },           OperSpec::NoStretch }, // "||"
	{ "}"		    	    	,		Mml::PostfixForm, {			    0,  		"true", 		      0,		   "0em",			0,			 0,		      "0em",			   0,		       "true" },            OperSpec::VStretch }, // "}"
	{ "~"		    	    	,		  Mml::InfixForm, {			    0,  		     0, 		      0,     "verythinmathspace",			0,			 0,	"verythinmathspace",			   0,			    0 },           OperSpec::NoStretch }, // "~"

	{  0                            ,                 Mml::InfixForm, { 	    	            0,                       0,                       0,            	       0,                     	0, 		         0, 	    	    	  0,                 	   0, 	              	    0 },           OperSpec::NoStretch }
};

static const OperSpec g_oper_spec_defaults =
	{  0                            ,                 Mml::InfixForm, { 	    	      "false",                 "false",                 "false",         "thickmathspace",                     "1", 		   "false", 	   "thickmathspace",                 "false", 	              "false" },           OperSpec::NoStretch  };

static const uint g_oper_spec_count = sizeof(g_oper_spec_data)/sizeof(OperSpec) - 1;

static const EntitySpec g_xml_entity_data[] = {
    { "angzarr",	"&#x0237C;" },
    { "cirmid",	"&#x02AEF;" },
    { "cudarrl",	"&#x02938;" },
    { "cudarrr",	"&#x02935;" },
    { "cularr",	"&#x021B6;" },
    { "cularrp",	"&#x0293D;" },
    { "curarr",	"&#x021B7;" },
    { "curarrm",	"&#x0293C;" },
    { "Darr",	"&#x021A1;" },
    { "dArr",	"&#x021D3;" },
    { "ddarr",	"&#x021CA;" },
    { "DDotrahd",	"&#x02911;" },
    { "dfisht",	"&#x0297F;" },
    { "dHar",	"&#x02965;" },
    { "dharl",	"&#x021C3;" },
    { "dharr",	"&#x021C2;" },
    { "duarr",	"&#x021F5;" },
    { "duhar",	"&#x0296F;" },
    { "dzigrarr",	"&#x027FF;" },
    { "erarr",	"&#x02971;" },
    { "hArr",	"&#x021D4;" },
    { "harr",	"&#x02194;" },
    { "harrcir",	"&#x02948;" },
    { "harrw",	"&#x021AD;" },
    { "hoarr",	"&#x021FF;" },
    { "imof",	"&#x022B7;" },
    { "lAarr",	"&#x021DA;" },
    { "Larr",	"&#x0219E;" },
    { "larrbfs",	"&#x0291F;" },
    { "larrfs",	"&#x0291D;" },
    { "larrhk",	"&#x021A9;" },
    { "larrlp",	"&#x021AB;" },
    { "larrpl",	"&#x02939;" },
    { "larrsim",	"&#x02973;" },
    { "larrtl",	"&#x021A2;" },
    { "lAtail",	"&#x0291B;" },
    { "latail",	"&#x02919;" },
    { "lBarr",	"&#x0290E;" },
    { "lbarr",	"&#x0290C;" },
    { "ldca",	"&#x02936;" },
    { "ldrdhar",	"&#x02967;" },
    { "ldrushar",	"&#x0294B;" },
    { "ldsh",	"&#x021B2;" },
    { "lfisht",	"&#x0297C;" },
    { "lHar",	"&#x02962;" },
    { "lhard",	"&#x021BD;" },
    { "lharu",	"&#x021BC;" },
    { "lharul",	"&#x0296A;" },
    { "llarr",	"&#x021C7;" },
    { "llhard",	"&#x0296B;" },
    { "loarr",	"&#x021FD;" },
    { "lrarr",	"&#x021C6;" },
    { "lrhar",	"&#x021CB;" },
    { "lrhard",	"&#x0296D;" },
    { "lsh",	"&#x021B0;" },
    { "lurdshar",	"&#x0294A;" },
    { "luruhar",	"&#x02966;" },
    { "Map",	"&#x02905;" },
    { "map",	"&#x021A6;" },
    { "midcir",	"&#x02AF0;" },
    { "mumap",	"&#x022B8;" },
    { "nearhk",	"&#x02924;" },
    { "neArr",	"&#x021D7;" },
    { "nearr",	"&#x02197;" },
    { "nesear",	"&#x02928;" },
    { "nhArr",	"&#x021CE;" },
    { "nharr",	"&#x021AE;" },
    { "nlArr",	"&#x021CD;" },
    { "nlarr",	"&#x0219A;" },
    { "nrArr",	"&#x021CF;" },
    { "nrarr",	"&#x0219B;" },
    { "nrarrc",	"&#x02933;&#x00338;" },
    { "nrarrw",	"&#x0219D;&#x00338;" },
    { "nvHarr",	"&#x02904;" },
    { "nvlArr",	"&#x02902;" },
    { "nvrArr",	"&#x02903;" },
    { "nwarhk",	"&#x02923;" },
    { "nwArr",	"&#x021D6;" },
    { "nwarr",	"&#x02196;" },
    { "nwnear",	"&#x02927;" },
    { "olarr",	"&#x021BA;" },
    { "orarr",	"&#x021BB;" },
    { "origof",	"&#x022B6;" },
    { "rAarr",	"&#x021DB;" },
    { "Rarr",	"&#x021A0;" },
    { "rarrap",	"&#x02975;" },
    { "rarrbfs",	"&#x02920;" },
    { "rarrc",	"&#x02933;" },
    { "rarrfs",	"&#x0291E;" },
    { "rarrhk",	"&#x021AA;" },
    { "rarrlp",	"&#x021AC;" },
    { "rarrpl",	"&#x02945;" },
    { "rarrsim",	"&#x02974;" },
    { "Rarrtl",	"&#x02916;" },
    { "rarrtl",	"&#x021A3;" },
    { "rarrw",	"&#x0219D;" },
    { "rAtail",	"&#x0291C;" },
    { "ratail",	"&#x0291A;" },
    { "RBarr",	"&#x02910;" },
    { "rBarr",	"&#x0290F;" },
    { "rbarr",	"&#x0290D;" },
    { "rdca",	"&#x02937;" },
    { "rdldhar",	"&#x02969;" },
    { "rdsh",	"&#x021B3;" },
    { "rfisht",	"&#x0297D;" },
    { "rHar",	"&#x02964;" },
    { "rhard",	"&#x021C1;" },
    { "rharu",	"&#x021C0;" },
    { "rharul",	"&#x0296C;" },
    { "rlarr",	"&#x021C4;" },
    { "rlhar",	"&#x021CC;" },
    { "roarr",	"&#x021FE;" },
    { "rrarr",	"&#x021C9;" },
    { "rsh",	"&#x021B1;" },
    { "ruluhar",	"&#x02968;" },
    { "searhk",	"&#x02925;" },
    { "seArr",	"&#x021D8;" },
    { "searr",	"&#x02198;" },
    { "seswar",	"&#x02929;" },
    { "simrarr",	"&#x02972;" },
    { "slarr",	"&#x02190;" },
    { "srarr",	"&#x02192;" },
    { "swarhk",	"&#x02926;" },
    { "swArr",	"&#x021D9;" },
    { "swarr",	"&#x02199;" },
    { "swnwar",	"&#x0292A;" },
    { "Uarr",	"&#x0219F;" },
    { "uArr",	"&#x021D1;" },
    { "Uarrocir",	"&#x02949;" },
    { "udarr",	"&#x021C5;" },
    { "udhar",	"&#x0296E;" },
    { "ufisht",	"&#x0297E;" },
    { "uHar",	"&#x02963;" },
    { "uharl",	"&#x021BF;" },
    { "uharr",	"&#x021BE;" },
    { "uuarr",	"&#x021C8;" },
    { "vArr",	"&#x021D5;" },
    { "varr",	"&#x02195;" },
    { "xhArr",	"&#x027FA;" },
    { "xharr",	"&#x027F7;" },
    { "xlArr",	"&#x027F8;" },
    { "xlarr",	"&#x027F5;" },
    { "xmap",	"&#x027FC;" },
    { "xrArr",	"&#x027F9;" },
    { "xrarr",	"&#x027F6;" },
    { "zigrarr",	"&#x021DD;" },
    { "ac",	"&#x0223E;" },
    { "acE",	"&#x0223E;&#x00333;" },
    { "amalg",	"&#x02A3F;" },
    { "barvee",	"&#x022BD;" },
    { "Barwed",	"&#x02306;" },
    { "barwed",	"&#x02305;" },
    { "bsolb",	"&#x029C5;" },
    { "Cap",	"&#x022D2;" },
    { "capand",	"&#x02A44;" },
    { "capbrcup",	"&#x02A49;" },
    { "capcap",	"&#x02A4B;" },
    { "capcup",	"&#x02A47;" },
    { "capdot",	"&#x02A40;" },
    { "caps",	"&#x02229;&#x0FE00;" },
    { "ccaps",	"&#x02A4D;" },
    { "ccups",	"&#x02A4C;" },
    { "ccupssm",	"&#x02A50;" },
    { "coprod",	"&#x02210;" },
    { "Cup",	"&#x022D3;" },
    { "cupbrcap",	"&#x02A48;" },
    { "cupcap",	"&#x02A46;" },
    { "cupcup",	"&#x02A4A;" },
    { "cupdot",	"&#x0228D;" },
    { "cupor",	"&#x02A45;" },
    { "cups",	"&#x0222A;&#x0FE00;" },
    { "cuvee",	"&#x022CE;" },
    { "cuwed",	"&#x022CF;" },
    { "Dagger",	"&#x02021;" },
    { "dagger",	"&#x02020;" },
    { "diam",	"&#x022C4;" },
    { "divonx",	"&#x022C7;" },
    { "eplus",	"&#x02A71;" },
    { "hercon",	"&#x022B9;" },
    { "intcal",	"&#x022BA;" },
    { "iprod",	"&#x02A3C;" },
    { "loplus",	"&#x02A2D;" },
    { "lotimes",	"&#x02A34;" },
    { "lthree",	"&#x022CB;" },
    { "ltimes",	"&#x022C9;" },
    { "midast",	"&#x0002A;" },
    { "minusb",	"&#x0229F;" },
    { "minusd",	"&#x02238;" },
    { "minusdu",	"&#x02A2A;" },
    { "ncap",	"&#x02A43;" },
    { "ncup",	"&#x02A42;" },
    { "oast",	"&#x0229B;" },
    { "ocir",	"&#x0229A;" },
    { "odash",	"&#x0229D;" },
    { "odiv",	"&#x02A38;" },
    { "odot",	"&#x02299;" },
    { "odsold",	"&#x029BC;" },
    { "ofcir",	"&#x029BF;" },
    { "ogt",	"&#x029C1;" },
    { "ohbar",	"&#x029B5;" },
    { "olcir",	"&#x029BE;" },
    { "olt",	"&#x029C0;" },
    { "omid",	"&#x029B6;" },
    { "ominus",	"&#x02296;" },
    { "opar",	"&#x029B7;" },
    { "operp",	"&#x029B9;" },
    { "oplus",	"&#x02295;" },
    { "osol",	"&#x02298;" },
    { "Otimes",	"&#x02A37;" },
    { "otimes",	"&#x02297;" },
    { "otimesas",	"&#x02A36;" },
    { "ovbar",	"&#x0233D;" },
    { "plusacir",	"&#x02A23;" },
    { "plusb",	"&#x0229E;" },
    { "pluscir",	"&#x02A22;" },
    { "plusdo",	"&#x02214;" },
    { "plusdu",	"&#x02A25;" },
    { "pluse",	"&#x02A72;" },
    { "plussim",	"&#x02A26;" },
    { "plustwo",	"&#x02A27;" },
    { "prod",	"&#x0220F;" },
    { "race",	"&#x029DA;" },
    { "roplus",	"&#x02A2E;" },
    { "rotimes",	"&#x02A35;" },
    { "rthree",	"&#x022CC;" },
    { "rtimes",	"&#x022CA;" },
    { "sdot",	"&#x022C5;" },
    { "sdotb",	"&#x022A1;" },
    { "setmn",	"&#x02216;" },
    { "simplus",	"&#x02A24;" },
    { "smashp",	"&#x02A33;" },
    { "solb",	"&#x029C4;" },
    { "sqcap",	"&#x02293;" },
    { "sqcaps",	"&#x02293;&#x0FE00;" },
    { "sqcup",	"&#x02294;" },
    { "sqcups",	"&#x02294;&#x0FE00;" },
    { "ssetmn",	"&#x02216;" },
    { "sstarf",	"&#x022C6;" },
    { "subdot",	"&#x02ABD;" },
    { "sum",	"&#x02211;" },
    { "supdot",	"&#x02ABE;" },
    { "timesb",	"&#x022A0;" },
    { "timesbar",	"&#x02A31;" },
    { "timesd",	"&#x02A30;" },
    { "tridot",	"&#x025EC;" },
    { "triminus",	"&#x02A3A;" },
    { "triplus",	"&#x02A39;" },
    { "trisb",	"&#x029CD;" },
    { "tritime",	"&#x02A3B;" },
    { "uplus",	"&#x0228E;" },
    { "veebar",	"&#x022BB;" },
    { "wedbar",	"&#x02A5F;" },
    { "wreath",	"&#x02240;" },
    { "xcap",	"&#x022C2;" },
    { "xcirc",	"&#x025EF;" },
    { "xcup",	"&#x022C3;" },
    { "xdtri",	"&#x025BD;" },
    { "xodot",	"&#x02A00;" },
    { "xoplus",	"&#x02A01;" },
    { "xotime",	"&#x02A02;" },
    { "xsqcup",	"&#x02A06;" },
    { "xuplus",	"&#x02A04;" },
    { "xutri",	"&#x025B3;" },
    { "xvee",	"&#x022C1;" },
    { "xwedge",	"&#x022C0;" },
    { "dlcorn",	"&#x0231E;" },
    { "drcorn",	"&#x0231F;" },
    { "gtlPar",	"&#x02995;" },
    { "langd",	"&#x02991;" },
    { "lbrke",	"&#x0298B;" },
    { "lbrksld",	"&#x0298F;" },
    { "lbrkslu",	"&#x0298D;" },
    { "lceil",	"&#x02308;" },
    { "lfloor",	"&#x0230A;" },
    { "lmoust",	"&#x023B0;" },
    { "lparlt",	"&#x02993;" },
    { "ltrPar",	"&#x02996;" },
    { "rangd",	"&#x02992;" },
    { "rbrke",	"&#x0298C;" },
    { "rbrksld",	"&#x0298E;" },
    { "rbrkslu",	"&#x02990;" },
    { "rceil",	"&#x02309;" },
    { "rfloor",	"&#x0230B;" },
    { "rmoust",	"&#x023B1;" },
    { "rpargt",	"&#x02994;" },
    { "ulcorn",	"&#x0231C;" },
    { "urcorn",	"&#x0231D;" },
    { "gnap",	"&#x02A8A;" },
    { "gnE",	"&#x02269;" },
    { "gne",	"&#x02A88;" },
    { "gnsim",	"&#x022E7;" },
    { "gvnE",	"&#x02269;&#x0FE00;" },
    { "lnap",	"&#x02A89;" },
    { "lnE",	"&#x02268;" },
    { "lne",	"&#x02A87;" },
    { "lnsim",	"&#x022E6;" },
    { "lvnE",	"&#x02268;&#x0FE00;" },
    { "nap",	"&#x02249;" },
    { "napE",	"&#x02A70;&#x00338;" },
    { "napid",	"&#x0224B;&#x00338;" },
    { "ncong",	"&#x02247;" },
    { "ncongdot",	"&#x02A6D;&#x00338;" },
    { "nequiv",	"&#x02262;" },
    { "ngE",	"&#x02267;&#x00338;" },
    { "nge",	"&#x02271;" },
    { "nges",	"&#x02A7E;&#x00338;" },
    { "nGg",	"&#x022D9;&#x00338;" },
    { "ngsim",	"&#x02275;" },
    { "nGt",	"&#x0226B;&#x020D2;" },
    { "ngt",	"&#x0226F;" },
    { "nGtv",	"&#x0226B;&#x00338;" },
    { "nlE",	"&#x02266;&#x00338;" },
    { "nle",	"&#x02270;" },
    { "nles",	"&#x02A7D;&#x00338;" },
    { "nLl",	"&#x022D8;&#x00338;" },
    { "nlsim",	"&#x02274;" },
    { "nLt",	"&#x0226A;&#x020D2;" },
    { "nlt",	"&#x0226E;" },
    { "nltri",	"&#x022EA;" },
    { "nltrie",	"&#x022EC;" },
    { "nLtv",	"&#x0226A;&#x00338;" },
    { "nmid",	"&#x02224;" },
    { "npar",	"&#x02226;" },
    { "npr",	"&#x02280;" },
    { "nprcue",	"&#x022E0;" },
    { "npre",	"&#x02AAF;&#x00338;" },
    { "nrtri",	"&#x022EB;" },
    { "nrtrie",	"&#x022ED;" },
    { "nsc",	"&#x02281;" },
    { "nsccue",	"&#x022E1;" },
    { "nsce",	"&#x02AB0;&#x00338;" },
    { "nsim",	"&#x02241;" },
    { "nsime",	"&#x02244;" },
    { "nsmid",	"&#x02224;" },
    { "nspar",	"&#x02226;" },
    { "nsqsube",	"&#x022E2;" },
    { "nsqsupe",	"&#x022E3;" },
    { "nsub",	"&#x02284;" },
    { "nsubE",	"&#x02AC5;&#x0338;" },
    { "nsube",	"&#x02288;" },
    { "nsup",	"&#x02285;" },
    { "nsupE",	"&#x02AC6;&#x0338;" },
    { "nsupe",	"&#x02289;" },
    { "ntgl",	"&#x02279;" },
    { "ntlg",	"&#x02278;" },
    { "nvap",	"&#x0224D;&#x020D2;" },
    { "nVDash",	"&#x022AF;" },
    { "nVdash",	"&#x022AE;" },
    { "nvDash",	"&#x022AD;" },
    { "nvdash",	"&#x022AC;" },
    { "nvge",	"&#x02265;&#x020D2;" },
    { "nvgt",	"&#x0003E;&#x020D2;" },
    { "nvle",	"&#x02264;&#x020D2;" },
    { "nvlt",	"&#x0003C;&#x020D2;" },
    { "nvltrie",	"&#x022B4;&#x020D2;" },
    { "nvrtrie",	"&#x022B5;&#x020D2;" },
    { "nvsim",	"&#x0223C;&#x020D2;" },
    { "parsim",	"&#x02AF3;" },
    { "prnap",	"&#x02AB9;" },
    { "prnE",	"&#x02AB5;" },
    { "prnsim",	"&#x022E8;" },
    { "rnmid",	"&#x02AEE;" },
    { "scnap",	"&#x02ABA;" },
    { "scnE",	"&#x02AB6;" },
    { "scnsim",	"&#x022E9;" },
    { "simne",	"&#x02246;" },
    { "solbar",	"&#x0233F;" },
    { "subnE",	"&#x02ACB;" },
    { "subne",	"&#x0228A;" },
    { "supnE",	"&#x02ACC;" },
    { "supne",	"&#x0228B;" },
    { "vnsub",	"&#x02282;&#x020D2;" },
    { "vnsup",	"&#x02283;&#x020D2;" },
    { "vsubnE",	"&#x02ACB;&#x0FE00;" },
    { "vsubne",	"&#x0228A;&#x0FE00;" },
    { "vsupnE",	"&#x02ACC;&#x0FE00;" },
    { "vsupne",	"&#x0228B;&#x0FE00;" },
    { "ang",	"&#x02220;" },
    { "ange",	"&#x029A4;" },
    { "angmsd",	"&#x02221;" },
    { "angmsdaa",	"&#x029A8;" },
    { "angmsdab",	"&#x029A9;" },
    { "angmsdac",	"&#x029AA;" },
    { "angmsdad",	"&#x029AB;" },
    { "angmsdae",	"&#x029AC;" },
    { "angmsdaf",	"&#x029AD;" },
    { "angmsdag",	"&#x029AE;" },
    { "angmsdah",	"&#x029AF;" },
    { "angrtvb",	"&#x022BE;" },
    { "angrtvbd",	"&#x0299D;" },
    { "bbrk",	"&#x023B5;" },
    { "bemptyv",	"&#x029B0;" },
    { "beth",	"&#x02136;" },
    { "boxbox",	"&#x029C9;" },
    { "bprime",	"&#x02035;" },
    { "bsemi",	"&#x0204F;" },
    { "cemptyv",	"&#x029B2;" },
    { "cirE",	"&#x029C3;" },
    { "cirscir",	"&#x029C2;" },
    { "comp",	"&#x02201;" },
    { "daleth",	"&#x02138;" },
    { "demptyv",	"&#x029B1;" },
    { "ell",	"&#x02113;" },
    { "empty",	"&#x02205;" },
    { "emptyv",	"&#x02205;" },
    { "gimel",	"&#x02137;" },
    { "iiota",	"&#x02129;" },
    { "image",	"&#x02111;" },
    { "imath",	"&#x00131;" },
    { "jmath",	"&#x0006A;" },
    { "laemptyv",	"&#x029B4;" },
    { "lltri",	"&#x025FA;" },
    { "lrtri",	"&#x022BF;" },
    { "mho",	"&#x02127;" },
    { "nang",	"&#x02220;&#x020D2;" },
    { "nexist",	"&#x02204;" },
    { "oS",	"&#x024C8;" },
    { "planck",	"&#x0210F;" },
    { "plankv",	"&#x0210F;" },
    { "raemptyv",	"&#x029B3;" },
    { "range",	"&#x029A5;" },
    { "real",	"&#x0211C;" },
    { "tbrk",	"&#x023B4;" },
    { "ultri",	"&#x025F8;" },
    { "urtri",	"&#x025F9;" },
    { "vzigzag",	"&#x0299A;" },
    { "weierp",	"&#x02118;" },
    { "apE",	"&#x02A70;" },
    { "ape",	"&#x0224A;" },
    { "apid",	"&#x0224B;" },
    { "asymp",	"&#x02248;" },
    { "Barv",	"&#x02AE7;" },
    { "bcong",	"&#x0224C;" },
    { "bepsi",	"&#x003F6;" },
    { "bowtie",	"&#x022C8;" },
    { "bsim",	"&#x0223D;" },
    { "bsime",	"&#x022CD;" },
    { "bsolhsub",	"&#x0005C;&#x02282;" },
    { "bump",	"&#x0224E;" },
    { "bumpE",	"&#x02AAE;" },
    { "bumpe",	"&#x0224F;" },
    { "cire",	"&#x02257;" },
    { "Colon",	"&#x02237;" },
    { "Colone",	"&#x02A74;" },
    { "colone",	"&#x02254;" },
    { "congdot",	"&#x02A6D;" },
    { "csub",	"&#x02ACF;" },
    { "csube",	"&#x02AD1;" },
    { "csup",	"&#x02AD0;" },
    { "csupe",	"&#x02AD2;" },
    { "cuepr",	"&#x022DE;" },
    { "cuesc",	"&#x022DF;" },
    { "Dashv",	"&#x02AE4;" },
    { "dashv",	"&#x022A3;" },
    { "easter",	"&#x02A6E;" },
    { "ecir",	"&#x02256;" },
    { "ecolon",	"&#x02255;" },
    { "eDDot",	"&#x02A77;" },
    { "eDot",	"&#x02251;" },
    { "efDot",	"&#x02252;" },
    { "eg",	"&#x02A9A;" },
    { "egs",	"&#x02A96;" },
    { "egsdot",	"&#x02A98;" },
    { "el",	"&#x02A99;" },
    { "els",	"&#x02A95;" },
    { "elsdot",	"&#x02A97;" },
    { "equest",	"&#x0225F;" },
    { "equivDD",	"&#x02A78;" },
    { "erDot",	"&#x02253;" },
    { "esdot",	"&#x02250;" },
    { "Esim",	"&#x02A73;" },
    { "esim",	"&#x02242;" },
    { "fork",	"&#x022D4;" },
    { "forkv",	"&#x02AD9;" },
    { "frown",	"&#x02322;" },
    { "gap",	"&#x02A86;" },
    { "gE",	"&#x02267;" },
    { "gEl",	"&#x02A8C;" },
    { "gel",	"&#x022DB;" },
    { "ges",	"&#x02A7E;" },
    { "gescc",	"&#x02AA9;" },
    { "gesdot",	"&#x02A80;" },
    { "gesdoto",	"&#x02A82;" },
    { "gesdotol",	"&#x02A84;" },
    { "gesl",	"&#x022DB;&#x0FE00;" },
    { "gesles",	"&#x02A94;" },
    { "Gg",	"&#x022D9;" },
    { "gl",	"&#x02277;" },
    { "gla",	"&#x02AA5;" },
    { "glE",	"&#x02A92;" },
    { "glj",	"&#x02AA4;" },
    { "gsim",	"&#x02273;" },
    { "gsime",	"&#x02A8E;" },
    { "gsiml",	"&#x02A90;" },
    { "Gt",	"&#x0226B;" },
    { "gtcc",	"&#x02AA7;" },
    { "gtcir",	"&#x02A7A;" },
    { "gtdot",	"&#x022D7;" },
    { "gtquest",	"&#x02A7C;" },
    { "gtrarr",	"&#x02978;" },
    { "homtht",	"&#x0223B;" },
    { "lap",	"&#x02A85;" },
    { "lat",	"&#x02AAB;" },
    { "late",	"&#x02AAD;" },
    { "lates",	"&#x02AAD;&#x0FE00;" },
    { "lE",	"&#x02266;" },
    { "lEg",	"&#x02A8B;" },
    { "leg",	"&#x022DA;" },
    { "les",	"&#x02A7D;" },
    { "lescc",	"&#x02AA8;" },
    { "lesdot",	"&#x02A7F;" },
    { "lesdoto",	"&#x02A81;" },
    { "lesdotor",	"&#x02A83;" },
    { "lesg",	"&#x022DA;&#x0FE00;" },
    { "lesges",	"&#x02A93;" },
    { "lg",	"&#x02276;" },
    { "lgE",	"&#x02A91;" },
    { "Ll",	"&#x022D8;" },
    { "lsim",	"&#x02272;" },
    { "lsime",	"&#x02A8D;" },
    { "lsimg",	"&#x02A8F;" },
    { "Lt",	"&#x0226A;" },
    { "ltcc",	"&#x02AA6;" },
    { "ltcir",	"&#x02A79;" },
    { "ltdot",	"&#x022D6;" },
    { "ltlarr",	"&#x02976;" },
    { "ltquest",	"&#x02A7B;" },
    { "ltrie",	"&#x022B4;" },
    { "mcomma",	"&#x02A29;" },
    { "mDDot",	"&#x0223A;" },
    { "mid",	"&#x02223;" },
    { "mlcp",	"&#x02ADB;" },
    { "models",	"&#x022A7;" },
    { "mstpos",	"&#x0223E;" },
    { "Pr",	"&#x02ABB;" },
    { "pr",	"&#x0227A;" },
    { "prap",	"&#x02AB7;" },
    { "prcue",	"&#x0227C;" },
    { "prE",	"&#x02AB3;" },
    { "pre",	"&#x02AAF;" },
    { "prsim",	"&#x0227E;" },
    { "prurel",	"&#x022B0;" },
    { "ratio",	"&#x02236;" },
    { "rtrie",	"&#x022B5;" },
    { "rtriltri",	"&#x029CE;" },
    { "Sc",	"&#x02ABC;" },
    { "sc",	"&#x0227B;" },
    { "scap",	"&#x02AB8;" },
    { "sccue",	"&#x0227D;" },
    { "scE",	"&#x02AB4;" },
    { "sce",	"&#x02AB0;" },
    { "scsim",	"&#x0227F;" },
    { "sdote",	"&#x02A66;" },
    { "simg",	"&#x02A9E;" },
    { "simgE",	"&#x02AA0;" },
    { "siml",	"&#x02A9D;" },
    { "simlE",	"&#x02A9F;" },
    { "smid",	"&#x02223;" },
    { "smile",	"&#x02323;" },
    { "smt",	"&#x02AAA;" },
    { "smte",	"&#x02AAC;" },
    { "smtes",	"&#x02AAC;&#x0FE00;" },
    { "spar",	"&#x02225;" },
    { "sqsub",	"&#x0228F;" },
    { "sqsube",	"&#x02291;" },
    { "sqsup",	"&#x02290;" },
    { "sqsupe",	"&#x02292;" },
    { "Sub",	"&#x022D0;" },
    { "subE",	"&#x02AC5;" },
    { "subedot",	"&#x02AC3;" },
    { "submult",	"&#x02AC1;" },
    { "subplus",	"&#x02ABF;" },
    { "subrarr",	"&#x02979;" },
    { "subsim",	"&#x02AC7;" },
    { "subsub",	"&#x02AD5;" },
    { "subsup",	"&#x02AD3;" },
    { "Sup",	"&#x022D1;" },
    { "supdsub",	"&#x02AD8;" },
    { "supE",	"&#x02AC6;" },
    { "supedot",	"&#x02AC4;" },
    { "suphsol",	"&#x02283;&#x00338;" },
    { "suphsub",	"&#x02AD7;" },
    { "suplarr",	"&#x0297B;" },
    { "supmult",	"&#x02AC2;" },
    { "supplus",	"&#x02AC0;" },
    { "supsim",	"&#x02AC8;" },
    { "supsub",	"&#x02AD4;" },
    { "supsup",	"&#x02AD6;" },
    { "thkap",	"&#x02248;" },
    { "topfork",	"&#x02ADA;" },
    { "trie",	"&#x0225C;" },
    { "twixt",	"&#x0226C;" },
    { "Vbar",	"&#x02AEB;" },
    { "vBar",	"&#x02AE8;" },
    { "vBarv",	"&#x02AE9;" },
    { "VDash",	"&#x022AB;" },
    { "Vdash",	"&#x022A9;" },
    { "vDash",	"&#x022A8;" },
    { "vdash",	"&#x022A2;" },
    { "Vdashl",	"&#x02AE6;" },
    { "vltri",	"&#x022B2;" },
    { "vprop",	"&#x0221D;" },
    { "vrtri",	"&#x022B3;" },
    { "Vvdash",	"&#x022AA;" },
    { "alpha",	"&#x003B1;" },
    { "beta",	"&#x003B2;" },
    { "chi",	"&#x003C7;" },
    { "Delta",	"&#x00394;" },
    { "delta",	"&#x003B4;" },
    { "epsi",	"&#x003B5;" },
    { "epsiv",	"&#x0025B;" },
    { "eta",	"&#x003B7;" },
    { "Gamma",	"&#x00393;" },
    { "gamma",	"&#x003B3;" },
    { "Gammad",	"&#x003DC;" },
    { "gammad",	"&#x003DD;" },
    { "iota",	"&#x003B9;" },
    { "kappa",	"&#x003BA;" },
    { "kappav",	"&#x003F0;" },
    { "Lambda",	"&#x0039B;" },
    { "lambda",	"&#x003BB;" },
    { "mu",	"&#x003BC;" },
    { "nu",	"&#x003BD;" },
    { "Omega",	"&#x003A9;" },
    { "omega",	"&#x003C9;" },
    { "Phi",	"&#x003A6;" },
    { "phi",	"&#x003D5;" },
    { "phiv",	"&#x003C6;" },
    { "Pi",	"&#x003A0;" },
    { "pi",	"&#x003C0;" },
    { "piv",	"&#x003D6;" },
    { "Psi",	"&#x003A8;" },
    { "psi",	"&#x003C8;" },
    { "rho",	"&#x003C1;" },
    { "rhov",	"&#x003F1;" },
    { "Sigma",	"&#x003A3;" },
    { "sigma",	"&#x003C3;" },
    { "sigmav",	"&#x003C2;" },
    { "tau",	"&#x003C4;" },
    { "Theta",	"&#x00398;" },
    { "theta",	"&#x003B8;" },
    { "thetav",	"&#x003D1;" },
    { "Upsi",	"&#x003D2;" },
    { "upsi",	"&#x003C5;" },
    { "Xi",	"&#x0039E;" },
    { "xi",	"&#x003BE;" },
    { "zeta",	"&#x003B6;" },
    { "Cfr",	"&#x0212D;" },
    { "Hfr",	"&#x0210C;" },
    { "Ifr",	"&#x02111;" },
    { "Rfr",	"&#x0211C;" },
    { "Zfr",	"&#x02128;" },
    { "Copf",	"&#x02102;" },
    { "Hopf",	"&#x0210D;" },
    { "Nopf",	"&#x02115;" },
    { "Popf",	"&#x02119;" },
    { "Qopf",	"&#x0211A;" },
    { "Ropf",	"&#x0211D;" },
    { "Zopf",	"&#x02124;" },
    { "Bscr",	"&#x0212C;" },
    { "Escr",	"&#x02130;" },
    { "escr",	"&#x0212F;" },
    { "Fscr",	"&#x02131;" },
    { "gscr",	"&#x0210A;" },
    { "Hscr",	"&#x0210B;" },
    { "Iscr",	"&#x02110;" },
    { "Lscr",	"&#x02112;" },
    { "Mscr",	"&#x02133;" },
    { "oscr",	"&#x02134;" },
    { "pscr",	"&#x1D4C5;" },
    { "Rscr",	"&#x0211B;" },
    { "acd",	"&#x0223F;" },
    { "aleph",	"&#x02135;" },
    { "And",	"&#x02A53;" },
    { "and",	"&#x02227;" },
    { "andand",	"&#x02A55;" },
    { "andd",	"&#x02A5C;" },
    { "andslope",	"&#x02A58;" },
    { "andv",	"&#x02A5A;" },
    { "angrt",	"&#x0221F;" },
    { "angsph",	"&#x02222;" },
    { "angst",	"&#x0212B;" },
    { "ap",	"&#x02248;" },
    { "apacir",	"&#x02A6F;" },
    { "awconint",	"&#x02233;" },
    { "awint",	"&#x02A11;" },
    { "becaus",	"&#x02235;" },
    { "bernou",	"&#x0212C;" },
    { "bne",	"&#x0003D;&#x020E5;" },
    { "bnequiv",	"&#x02261;&#x020E5;" },
    { "bNot",	"&#x02AED;" },
    { "bnot",	"&#x02310;" },
    { "bottom",	"&#x022A5;" },
    { "cap",	"&#x02229;" },
    { "Cconint",	"&#x02230;" },
    { "cirfnint",	"&#x02A10;" },
    { "compfn",	"&#x02218;" },
    { "cong",	"&#x02245;" },
    { "Conint",	"&#x0222F;" },
    { "conint",	"&#x0222E;" },
    { "ctdot",	"&#x022EF;" },
    { "cup",	"&#x0222A;" },
    { "cwconint",	"&#x02232;" },
    { "cwint",	"&#x02231;" },
    { "cylcty",	"&#x0232D;" },
    { "disin",	"&#x022F2;" },
    { "Dot",	"&#x000A8;" },
    { "DotDot",	"&#x020DC;" },
    { "dsol",	"&#x029F6;" },
    { "dtdot",	"&#x022F1;" },
    { "dwangle",	"&#x029A6;" },
    { "epar",	"&#x022D5;" },
    { "eparsl",	"&#x029E3;" },
    { "equiv",	"&#x02261;" },
    { "eqvparsl",	"&#x029E5;" },
    { "exist",	"&#x02203;" },
    { "fnof",	"&#x00192;" },
    { "forall",	"&#x02200;" },
    { "fpartint",	"&#x02A0D;" },
    { "ge",	"&#x02265;" },
    { "hamilt",	"&#x0210B;" },
    { "iff",	"&#x021D4;" },
    { "iinfin",	"&#x029DC;" },
    { "infin",	"&#x0221E;" },
    { "Int",	"&#x0222C;" },
    { "int",	"&#x0222B;" },
    { "intlarhk",	"&#x02A17;" },
    { "isin",	"&#x02208;" },
    { "isindot",	"&#x022F5;" },
    { "isinE",	"&#x022F9;" },
    { "isins",	"&#x022F4;" },
    { "isinsv",	"&#x022F3;" },
    { "isinv",	"&#x02208;" },
    { "lagran",	"&#x02112;" },
    { "Lang",	"&#x0300A;" },
    { "lang",	"&#x02329;" },
    { "lArr",	"&#x021D0;" },
    { "lbbrk",	"&#x03014;" },
    { "le",	"&#x02264;" },
    { "loang",	"&#x03018;" },
    { "lobrk",	"&#x0301A;" },
    { "lopar",	"&#x02985;" },
    { "lowast",	"&#x02217;" },
    { "minus",	"&#x02212;" },
    { "mnplus",	"&#x02213;" },
    { "nabla",	"&#x02207;" },
    { "ne",	"&#x02260;" },
    { "nedot",	"&#x02250;&#x00338;" },
    { "nhpar",	"&#x02AF2;" },
    { "ni",	"&#x0220B;" },
    { "nis",	"&#x022FC;" },
    { "nisd",	"&#x022FA;" },
    { "niv",	"&#x0220B;" },
    { "Not",	"&#x02AEC;" },
    { "notin",	"&#x02209;" },
    { "notindot",	"&#x022F5;&#x00338;" },
    { "notinva",	"&#x02209;" },
    { "notinvb",	"&#x022F7;" },
    { "notinvc",	"&#x022F6;" },
    { "notni",	"&#x0220C;" },
    { "notniva",	"&#x0220C;" },
    { "notnivb",	"&#x022FE;" },
    { "notnivc",	"&#x022FD;" },
    { "nparsl",	"&#x02AFD;&#x020E5;" },
    { "npart",	"&#x02202;&#x00338;" },
    { "npolint",	"&#x02A14;" },
    { "nvinfin",	"&#x029DE;" },
    { "olcross",	"&#x029BB;" },
    { "Or",	"&#x02A54;" },
    { "or",	"&#x02228;" },
    { "ord",	"&#x02A5D;" },
    { "order",	"&#x02134;" },
    { "oror",	"&#x02A56;" },
    { "orslope",	"&#x02A57;" },
    { "orv",	"&#x02A5B;" },
    { "par",	"&#x02225;" },
    { "parsl",	"&#x02AFD;" },
    { "part",	"&#x02202;" },
    { "permil",	"&#x02030;" },
    { "perp",	"&#x022A5;" },
    { "pertenk",	"&#x02031;" },
    { "phmmat",	"&#x02133;" },
    { "pointint",	"&#x02A15;" },
    { "Prime",	"&#x02033;" },
    { "prime",	"&#x02032;" },
    { "profalar",	"&#x0232E;" },
    { "profline",	"&#x02312;" },
    { "profsurf",	"&#x02313;" },
    { "prop",	"&#x0221D;" },
    { "qint",	"&#x02A0C;" },
    { "qprime",	"&#x02057;" },
    { "quatint",	"&#x02A16;" },
    { "radic",	"&#x0221A;" },
    { "Rang",	"&#x0300B;" },
    { "rang",	"&#x0232A;" },
    { "rArr",	"&#x021D2;" },
    { "rbbrk",	"&#x03015;" },
    { "roang",	"&#x03019;" },
    { "robrk",	"&#x0301B;" },
    { "ropar",	"&#x02986;" },
    { "rppolint",	"&#x02A12;" },
    { "scpolint",	"&#x02A13;" },
    { "sim",	"&#x0223C;" },
    { "simdot",	"&#x02A6A;" },
    { "sime",	"&#x02243;" },
    { "smeparsl",	"&#x029E4;" },
    { "square",	"&#x025A1;" },
    { "squarf",	"&#x025AA;" },
    { "sub",	"&#x02282;" },
    { "sube",	"&#x02286;" },
    { "sup",	"&#x02283;" },
    { "supe",	"&#x02287;" },
    { "tdot",	"&#x020DB;" },
    { "there4",	"&#x02234;" },
    { "tint",	"&#x0222D;" },
    { "top",	"&#x022A4;" },
    { "topbot",	"&#x02336;" },
    { "topcir",	"&#x02AF1;" },
    { "tprime",	"&#x02034;" },
    { "utdot",	"&#x022F0;" },
    { "uwangle",	"&#x029A7;" },
    { "vangrt",	"&#x0299C;" },
    { "veeeq",	"&#x0225A;" },
    { "Verbar",	"&#x02016;" },
    { "wedgeq",	"&#x02259;" },
    { "xnis",	"&#x022FB;" },
    { "boxDL",	"&#x02557;" },
    { "boxDl",	"&#x02556;" },
    { "boxdL",	"&#x02555;" },
    { "boxdl",	"&#x02510;" },
    { "boxDR",	"&#x02554;" },
    { "boxDr",	"&#x02553;" },
    { "boxdR",	"&#x02552;" },
    { "boxdr",	"&#x0250C;" },
    { "boxH",	"&#x02550;" },
    { "boxh",	"&#x02500;" },
    { "boxHD",	"&#x02566;" },
    { "boxHd",	"&#x02564;" },
    { "boxhD",	"&#x02565;" },
    { "boxhd",	"&#x0252C;" },
    { "boxHU",	"&#x02569;" },
    { "boxHu",	"&#x02567;" },
    { "boxhU",	"&#x02568;" },
    { "boxhu",	"&#x02534;" },
    { "boxUL",	"&#x0255D;" },
    { "boxUl",	"&#x0255C;" },
    { "boxuL",	"&#x0255B;" },
    { "boxul",	"&#x02518;" },
    { "boxUR",	"&#x0255A;" },
    { "boxUr",	"&#x02559;" },
    { "boxuR",	"&#x02558;" },
    { "boxur",	"&#x02514;" },
    { "boxV",	"&#x02551;" },
    { "boxv",	"&#x02502;" },
    { "boxVH",	"&#x0256C;" },
    { "boxVh",	"&#x0256B;" },
    { "boxvH",	"&#x0256A;" },
    { "boxvh",	"&#x0253C;" },
    { "boxVL",	"&#x02563;" },
    { "boxVl",	"&#x02562;" },
    { "boxvL",	"&#x02561;" },
    { "boxvl",	"&#x02524;" },
    { "boxVR",	"&#x02560;" },
    { "boxVr",	"&#x0255F;" },
    { "boxvR",	"&#x0255E;" },
    { "boxvr",	"&#x0251C;" },
    { "Acy",	"&#x00410;" },
    { "acy",	"&#x00430;" },
    { "Bcy",	"&#x00411;" },
    { "bcy",	"&#x00431;" },
    { "CHcy",	"&#x00427;" },
    { "chcy",	"&#x00447;" },
    { "Dcy",	"&#x00414;" },
    { "dcy",	"&#x00434;" },
    { "Ecy",	"&#x0042D;" },
    { "ecy",	"&#x0044D;" },
    { "Fcy",	"&#x00424;" },
    { "fcy",	"&#x00444;" },
    { "Gcy",	"&#x00413;" },
    { "gcy",	"&#x00433;" },
    { "HARDcy",	"&#x0042A;" },
    { "hardcy",	"&#x0044A;" },
    { "Icy",	"&#x00418;" },
    { "icy",	"&#x00438;" },
    { "IEcy",	"&#x00415;" },
    { "iecy",	"&#x00435;" },
    { "IOcy",	"&#x00401;" },
    { "iocy",	"&#x00451;" },
    { "Jcy",	"&#x00419;" },
    { "jcy",	"&#x00439;" },
    { "Kcy",	"&#x0041A;" },
    { "kcy",	"&#x0043A;" },
    { "KHcy",	"&#x00425;" },
    { "khcy",	"&#x00445;" },
    { "Lcy",	"&#x0041B;" },
    { "lcy",	"&#x0043B;" },
    { "Mcy",	"&#x0041C;" },
    { "mcy",	"&#x0043C;" },
    { "Ncy",	"&#x0041D;" },
    { "ncy",	"&#x0043D;" },
    { "numero",	"&#x02116;" },
    { "Ocy",	"&#x0041E;" },
    { "ocy",	"&#x0043E;" },
    { "Pcy",	"&#x0041F;" },
    { "pcy",	"&#x0043F;" },
    { "Rcy",	"&#x00420;" },
    { "rcy",	"&#x00440;" },
    { "Scy",	"&#x00421;" },
    { "scy",	"&#x00441;" },
    { "SHCHcy",	"&#x00429;" },
    { "shchcy",	"&#x00449;" },
    { "SHcy",	"&#x00428;" },
    { "shcy",	"&#x00448;" },
    { "SOFTcy",	"&#x0042C;" },
    { "softcy",	"&#x0044C;" },
    { "Tcy",	"&#x00422;" },
    { "tcy",	"&#x00442;" },
    { "TScy",	"&#x00426;" },
    { "tscy",	"&#x00446;" },
    { "Ucy",	"&#x00423;" },
    { "ucy",	"&#x00443;" },
    { "Vcy",	"&#x00412;" },
    { "vcy",	"&#x00432;" },
    { "YAcy",	"&#x0042F;" },
    { "yacy",	"&#x0044F;" },
    { "Ycy",	"&#x0042B;" },
    { "ycy",	"&#x0044B;" },
    { "YUcy",	"&#x0042E;" },
    { "yucy",	"&#x0044E;" },
    { "Zcy",	"&#x00417;" },
    { "zcy",	"&#x00437;" },
    { "ZHcy",	"&#x00416;" },
    { "zhcy",	"&#x00436;" },
    { "DJcy",	"&#x00402;" },
    { "djcy",	"&#x00452;" },
    { "DScy",	"&#x00405;" },
    { "dscy",	"&#x00455;" },
    { "DZcy",	"&#x0040F;" },
    { "dzcy",	"&#x0045F;" },
    { "GJcy",	"&#x00403;" },
    { "gjcy",	"&#x00453;" },
    { "Iukcy",	"&#x00406;" },
    { "iukcy",	"&#x00456;" },
    { "Jsercy",	"&#x00408;" },
    { "jsercy",	"&#x00458;" },
    { "Jukcy",	"&#x00404;" },
    { "jukcy",	"&#x00454;" },
    { "KJcy",	"&#x0040C;" },
    { "kjcy",	"&#x0045C;" },
    { "LJcy",	"&#x00409;" },
    { "ljcy",	"&#x00459;" },
    { "NJcy",	"&#x0040A;" },
    { "njcy",	"&#x0045A;" },
    { "TSHcy",	"&#x0040B;" },
    { "tshcy",	"&#x0045B;" },
    { "Ubrcy",	"&#x0040E;" },
    { "ubrcy",	"&#x0045E;" },
    { "YIcy",	"&#x00407;" },
    { "yicy",	"&#x00457;" },
    { "acute",	"&#x000B4;" },
    { "breve",	"&#x002D8;" },
    { "caron",	"&#x002C7;" },
    { "cedil",	"&#x000B8;" },
    { "circ",	"&#x002C6;" },
    { "dblac",	"&#x002DD;" },
    { "die",	"&#x000A8;" },
    { "dot",	"&#x002D9;" },
    { "grave",	"&#x00060;" },
    { "macr",	"&#x000AF;" },
    { "ogon",	"&#x002DB;" },
    { "ring",	"&#x002DA;" },
    { "tilde",	"&#x002DC;" },
    { "uml",	"&#x000A8;" },
    { "Aacute",	"&#x000C1;" },
    { "aacute",	"&#x000E1;" },
    { "Acirc",	"&#x000C2;" },
    { "acirc",	"&#x000E2;" },
    { "AElig",	"&#x000C6;" },
    { "aelig",	"&#x000E6;" },
    { "Agrave",	"&#x000C0;" },
    { "agrave",	"&#x000E0;" },
    { "Aring",	"&#x000C5;" },
    { "aring",	"&#x000E5;" },
    { "Atilde",	"&#x000C3;" },
    { "atilde",	"&#x000E3;" },
    { "Auml",	"&#x000C4;" },
    { "auml",	"&#x000E4;" },
    { "Ccedil",	"&#x000C7;" },
    { "ccedil",	"&#x000E7;" },
    { "Eacute",	"&#x000C9;" },
    { "eacute",	"&#x000E9;" },
    { "Ecirc",	"&#x000CA;" },
    { "ecirc",	"&#x000EA;" },
    { "Egrave",	"&#x000C8;" },
    { "egrave",	"&#x000E8;" },
    { "ETH",	"&#x000D0;" },
    { "eth",	"&#x000F0;" },
    { "Euml",	"&#x000CB;" },
    { "euml",	"&#x000EB;" },
    { "Iacute",	"&#x000CD;" },
    { "iacute",	"&#x000ED;" },
    { "Icirc",	"&#x000CE;" },
    { "icirc",	"&#x000EE;" },
    { "Igrave",	"&#x000CC;" },
    { "igrave",	"&#x000EC;" },
    { "Iuml",	"&#x000CF;" },
    { "iuml",	"&#x000EF;" },
    { "Ntilde",	"&#x000D1;" },
    { "ntilde",	"&#x000F1;" },
    { "Oacute",	"&#x000D3;" },
    { "oacute",	"&#x000F3;" },
    { "Ocirc",	"&#x000D4;" },
    { "ocirc",	"&#x000F4;" },
    { "Ograve",	"&#x000D2;" },
    { "ograve",	"&#x000F2;" },
    { "Oslash",	"&#x000D8;" },
    { "oslash",	"&#x000F8;" },
    { "Otilde",	"&#x000D5;" },
    { "otilde",	"&#x000F5;" },
    { "Ouml",	"&#x000D6;" },
    { "ouml",	"&#x000F6;" },
    { "szlig",	"&#x000DF;" },
    { "THORN",	"&#x000DE;" },
    { "thorn",	"&#x000FE;" },
    { "Uacute",	"&#x000DA;" },
    { "uacute",	"&#x000FA;" },
    { "Ucirc",	"&#x000DB;" },
    { "ucirc",	"&#x000FB;" },
    { "Ugrave",	"&#x000D9;" },
    { "ugrave",	"&#x000F9;" },
    { "Uuml",	"&#x000DC;" },
    { "uuml",	"&#x000FC;" },
    { "Yacute",	"&#x000DD;" },
    { "yacute",	"&#x000FD;" },
    { "yuml",	"&#x000FF;" },
    { "Abreve",	"&#x00102;" },
    { "abreve",	"&#x00103;" },
    { "Amacr",	"&#x00100;" },
    { "amacr",	"&#x00101;" },
    { "Aogon",	"&#x00104;" },
    { "aogon",	"&#x00105;" },
    { "Cacute",	"&#x00106;" },
    { "cacute",	"&#x00107;" },
    { "Ccaron",	"&#x0010C;" },
    { "ccaron",	"&#x0010D;" },
    { "Ccirc",	"&#x00108;" },
    { "ccirc",	"&#x00109;" },
    { "Cdot",	"&#x0010A;" },
    { "cdot",	"&#x0010B;" },
    { "Dcaron",	"&#x0010E;" },
    { "dcaron",	"&#x0010F;" },
    { "Dstrok",	"&#x00110;" },
    { "dstrok",	"&#x00111;" },
    { "Ecaron",	"&#x0011A;" },
    { "ecaron",	"&#x0011B;" },
    { "Edot",	"&#x00116;" },
    { "edot",	"&#x00117;" },
    { "Emacr",	"&#x00112;" },
    { "emacr",	"&#x00113;" },
    { "ENG",	"&#x0014A;" },
    { "eng",	"&#x0014B;" },
    { "Eogon",	"&#x00118;" },
    { "eogon",	"&#x00119;" },
    { "gacute",	"&#x001F5;" },
    { "Gbreve",	"&#x0011E;" },
    { "gbreve",	"&#x0011F;" },
    { "Gcedil",	"&#x00122;" },
    { "Gcirc",	"&#x0011C;" },
    { "gcirc",	"&#x0011D;" },
    { "Gdot",	"&#x00120;" },
    { "gdot",	"&#x00121;" },
    { "Hcirc",	"&#x00124;" },
    { "hcirc",	"&#x00125;" },
    { "Hstrok",	"&#x00126;" },
    { "hstrok",	"&#x00127;" },
    { "Idot",	"&#x00130;" },
    { "IJlig",	"&#x00132;" },
    { "ijlig",	"&#x00133;" },
    { "Imacr",	"&#x0012A;" },
    { "imacr",	"&#x0012B;" },
    { "inodot",	"&#x00131;" },
    { "Iogon",	"&#x0012E;" },
    { "iogon",	"&#x0012F;" },
    { "Itilde",	"&#x00128;" },
    { "itilde",	"&#x00129;" },
    { "Jcirc",	"&#x00134;" },
    { "jcirc",	"&#x00135;" },
    { "Kcedil",	"&#x00136;" },
    { "kcedil",	"&#x00137;" },
    { "kgreen",	"&#x00138;" },
    { "Lacute",	"&#x00139;" },
    { "lacute",	"&#x0013A;" },
    { "Lcaron",	"&#x0013D;" },
    { "lcaron",	"&#x0013E;" },
    { "Lcedil",	"&#x0013B;" },
    { "lcedil",	"&#x0013C;" },
    { "Lmidot",	"&#x0013F;" },
    { "lmidot",	"&#x00140;" },
    { "Lstrok",	"&#x00141;" },
    { "lstrok",	"&#x00142;" },
    { "Nacute",	"&#x00143;" },
    { "nacute",	"&#x00144;" },
    { "napos",	"&#x00149;" },
    { "Ncaron",	"&#x00147;" },
    { "ncaron",	"&#x00148;" },
    { "Ncedil",	"&#x00145;" },
    { "ncedil",	"&#x00146;" },
    { "Odblac",	"&#x00150;" },
    { "odblac",	"&#x00151;" },
    { "OElig",	"&#x00152;" },
    { "oelig",	"&#x00153;" },
    { "Omacr",	"&#x0014C;" },
    { "omacr",	"&#x0014D;" },
    { "Racute",	"&#x00154;" },
    { "racute",	"&#x00155;" },
    { "Rcaron",	"&#x00158;" },
    { "rcaron",	"&#x00159;" },
    { "Rcedil",	"&#x00156;" },
    { "rcedil",	"&#x00157;" },
    { "Sacute",	"&#x0015A;" },
    { "sacute",	"&#x0015B;" },
    { "Scaron",	"&#x00160;" },
    { "scaron",	"&#x00161;" },
    { "Scedil",	"&#x0015E;" },
    { "scedil",	"&#x0015F;" },
    { "Scirc",	"&#x0015C;" },
    { "scirc",	"&#x0015D;" },
    { "Tcaron",	"&#x00164;" },
    { "tcaron",	"&#x00165;" },
    { "Tcedil",	"&#x00162;" },
    { "tcedil",	"&#x00163;" },
    { "Tstrok",	"&#x00166;" },
    { "tstrok",	"&#x00167;" },
    { "Ubreve",	"&#x0016C;" },
    { "ubreve",	"&#x0016D;" },
    { "Udblac",	"&#x00170;" },
    { "udblac",	"&#x00171;" },
    { "Umacr",	"&#x0016A;" },
    { "umacr",	"&#x0016B;" },
    { "Uogon",	"&#x00172;" },
    { "uogon",	"&#x00173;" },
    { "Uring",	"&#x0016E;" },
    { "uring",	"&#x0016F;" },
    { "Utilde",	"&#x00168;" },
    { "utilde",	"&#x00169;" },
    { "Wcirc",	"&#x00174;" },
    { "wcirc",	"&#x00175;" },
    { "Ycirc",	"&#x00176;" },
    { "ycirc",	"&#x00177;" },
    { "Yuml",	"&#x00178;" },
    { "Zacute",	"&#x00179;" },
    { "zacute",	"&#x0017A;" },
    { "Zcaron",	"&#x0017D;" },
    { "zcaron",	"&#x0017E;" },
    { "Zdot",	"&#x0017B;" },
    { "zdot",	"&#x0017C;" },
    { "apos",	"&#x00027;" },
    { "ast",	"&#x0002A;" },
    { "brvbar",	"&#x000A6;" },
    { "bsol",	"&#x0005C;" },
    { "cent",	"&#x000A2;" },
    { "colon",	"&#x0003A;" },
    { "comma",	"&#x0002C;" },
    { "commat",	"&#x00040;" },
    { "copy",	"&#x000A9;" },
    { "curren",	"&#x000A4;" },
    { "darr",	"&#x02193;" },
    { "deg",	"&#x000B0;" },
    { "divide",	"&#x000F7;" },
    { "dollar",	"&#x00024;" },
    { "equals",	"&#x0003D;" },
    { "excl",	"&#x00021;" },
    { "frac12",	"&#x000BD;" },
    { "frac14",	"&#x000BC;" },
    { "frac18",	"&#x0215B;" },
    { "frac34",	"&#x000BE;" },
    { "frac38",	"&#x0215C;" },
    { "frac58",	"&#x0215D;" },
    { "frac78",	"&#x0215E;" },
    { "gt",	"&#x0003E;" },
    { "half",	"&#x000BD;" },
    { "horbar",	"&#x02015;" },
    { "hyphen",	"&#x02010;" },
    { "iexcl",	"&#x000A1;" },
    { "iquest",	"&#x000BF;" },
    { "laquo",	"&#x000AB;" },
    { "larr",	"&#x02190;" },
    { "lcub",	"&#x0007B;" },
    { "ldquo",	"&#x0201C;" },
    { "lowbar",	"&#x0005F;" },
    { "lpar",	"&#x00028;" },
    { "lsqb",	"&#x0005B;" },
    { "lsquo",	"&#x02018;" },
    { "lt",	"&#x0003C;" },
    { "micro",	"&#x000B5;" },
    { "middot",	"&#x000B7;" },
    { "nbsp",	"&#x000A0;" },
    { "not",	"&#x000AC;" },
    { "num",	"&#x00023;" },
    { "ohm",	"&#x02126;" },
    { "ordf",	"&#x000AA;" },
    { "ordm",	"&#x000BA;" },
    { "para",	"&#x000B6;" },
    { "percnt",	"&#x00025;" },
    { "period",	"&#x0002E;" },
    { "plus",	"&#x0002B;" },
    { "plusmn",	"&#x000B1;" },
    { "pound",	"&#x000A3;" },
    { "quest",	"&#x0003F;" },
    { "quot",	"&#x00022;" },
    { "raquo",	"&#x000BB;" },
    { "rarr",	"&#x02192;" },
    { "rcub",	"&#x0007D;" },
    { "rdquo",	"&#x0201D;" },
    { "reg",	"&#x000AE;" },
    { "rpar",	"&#x00029;" },
    { "rsqb",	"&#x0005D;" },
    { "rsquo",	"&#x02019;" },
    { "sect",	"&#x000A7;" },
    { "semi",	"&#x0003B;" },
    { "shy",	"&#x000AD;" },
    { "sol",	"&#x0002F;" },
    { "sung",	"&#x0266A;" },
    { "sup1",	"&#x000B9;" },
    { "sup2",	"&#x000B2;" },
    { "sup3",	"&#x000B3;" },
    { "times",	"&#x000D7;" },
    { "trade",	"&#x02122;" },
    { "uarr",	"&#x02191;" },
    { "verbar",	"&#x0007C;" },
    { "yen",	"&#x000A5;" },
    { "blank",	"&#x02423;" },
    { "blk12",	"&#x02592;" },
    { "blk14",	"&#x02591;" },
    { "blk34",	"&#x02593;" },
    { "block",	"&#x02588;" },
    { "bull",	"&#x02022;" },
    { "caret",	"&#x02041;" },
    { "check",	"&#x02713;" },
    { "cir",	"&#x025CB;" },
    { "clubs",	"&#x02663;" },
    { "copysr",	"&#x02117;" },
    { "cross",	"&#x02717;" },
    { "Dagger",	"&#x02021;" },
    { "dagger",	"&#x02020;" },
    { "dash",	"&#x02010;" },
    { "diams",	"&#x02666;" },
    { "dlcrop",	"&#x0230D;" },
    { "drcrop",	"&#x0230C;" },
    { "dtri",	"&#x025BF;" },
    { "dtrif",	"&#x025BE;" },
    { "emsp",	"&#x02003;" },
    { "emsp13",	"&#x02004;" },
    { "emsp14",	"&#x02005;" },
    { "ensp",	"&#x02002;" },
    { "female",	"&#x02640;" },
    { "ffilig",	"&#x0FB03;" },
    { "fflig",	"&#x0FB00;" },
    { "ffllig",	"&#x0FB04;" },
    { "filig",	"&#x0FB01;" },
    { "flat",	"&#x0266D;" },
    { "fllig",	"&#x0FB02;" },
    { "frac13",	"&#x02153;" },
    { "frac15",	"&#x02155;" },
    { "frac16",	"&#x02159;" },
    { "frac23",	"&#x02154;" },
    { "frac25",	"&#x02156;" },
    { "frac35",	"&#x02157;" },
    { "frac45",	"&#x02158;" },
    { "frac56",	"&#x0215A;" },
    { "hairsp",	"&#x0200A;" },
    { "hearts",	"&#x02665;" },
    { "hellip",	"&#x02026;" },
    { "hybull",	"&#x02043;" },
    { "incare",	"&#x02105;" },
    { "ldquor",	"&#x0201E;" },
    { "lhblk",	"&#x02584;" },
    { "loz",	"&#x025CA;" },
    { "lozf",	"&#x029EB;" },
    { "lsquor",	"&#x0201A;" },
    { "ltri",	"&#x025C3;" },
    { "ltrif",	"&#x025C2;" },
    { "male",	"&#x02642;" },
    { "malt",	"&#x02720;" },
    { "marker",	"&#x025AE;" },
    { "mdash",	"&#x02014;" },
    { "mldr",	"&#x02026;" },
    { "natur",	"&#x0266E;" },
    { "ndash",	"&#x02013;" },
    { "nldr",	"&#x02025;" },
    { "numsp",	"&#x02007;" },
    { "phone",	"&#x0260E;" },
    { "puncsp",	"&#x02008;" },
    { "rdquor",	"&#x0201D;" },
    { "rect",	"&#x025AD;" },
    { "rsquor",	"&#x02019;" },
    { "rtri",	"&#x025B9;" },
    { "rtrif",	"&#x025B8;" },
    { "rx",	"&#x0211E;" },
    { "sext",	"&#x02736;" },
    { "sharp",	"&#x0266F;" },
    { "spades",	"&#x02660;" },
    { "squ",	"&#x025A1;" },
    { "squf",	"&#x025AA;" },
    { "star",	"&#x02606;" },
    { "starf",	"&#x02605;" },
    { "target",	"&#x02316;" },
    { "telrec",	"&#x02315;" },
    { "thinsp",	"&#x02009;" },
    { "uhblk",	"&#x02580;" },
    { "ulcrop",	"&#x0230F;" },
    { "urcrop",	"&#x0230E;" },
    { "utri",	"&#x025B5;" },
    { "utrif",	"&#x025B4;" },
    { "vellip",	"&#x022EE;" },
    { "af",	"&#x02061;" },
    { "asympeq",	"&#x0224D;" },
    { "Cross",	"&#x02A2F;" },
    { "DD",	"&#x02145;" },
    { "dd",	"&#x02146;" },
    { "DownArrowBar",	"&#x02913;" },
    { "DownBreve",	"&#x00311;" },
    { "DownLeftRightVector",	"&#x02950;" },
    { "DownLeftTeeVector",	"&#x0295E;" },
    { "DownLeftVectorBar",	"&#x02956;" },
    { "DownRightTeeVector",	"&#x0295F;" },
    { "DownRightVectorBar",	"&#x02957;" },
    { "ee",	"&#x02147;" },
    { "EmptySmallSquare",	"&#x025FB;" },
    { "EmptyVerySmallSquare",	"&#x025AB;" },
    { "Equal",	"&#x02A75;" },
    { "FilledSmallSquare",	"&#x025FC;" },
    { "FilledVerySmallSquare",	"&#x025AA;" },
    { "GreaterGreater",	"&#x02AA2;" },
    { "Hat",	"&#x0005E;" },
    { "HorizontalLine",	"&#x02500;" },
    { "ic",	"&#x02063;" },
    { "ii",	"&#x02148;" },
    { "it",	"&#x02062;" },
    { "larrb",	"&#x021E4;" },
    { "LeftDownTeeVector",	"&#x02961;" },
    { "LeftDownVectorBar",	"&#x02959;" },
    { "LeftRightVector",	"&#x0294E;" },
    { "LeftTeeVector",	"&#x0295A;" },
    { "LeftTriangleBar",	"&#x029CF;" },
    { "LeftUpDownVector",	"&#x02951;" },
    { "LeftUpTeeVector",	"&#x02960;" },
    { "LeftUpVectorBar",	"&#x02958;" },
    { "LeftVectorBar",	"&#x02952;" },
    { "LessLess",	"&#x02AA1;" },
    { "mapstodown",	"&#x021A7;" },
    { "mapstoleft",	"&#x021A4;" },
    { "mapstoup",	"&#x021A5;" },
    { "MediumSpace",	"&#x0205F;" },
    { "nbump",	"&#x0224E;&#x00338;" },
    { "nbumpe",	"&#x0224F;&#x00338;" },
    { "nesim",	"&#x02242;&#x00338;" },
    { "NewLine",	"&#x0000A;" },
    { "NoBreak",	"&#x02060;" },
    { "NotCupCap",	"&#x0226D;" },
    { "NotHumpEqual",	"&#x0224F;&#x00338;" },
    { "NotLeftTriangleBar",	"&#x029CF;&#x00338;" },
    { "NotNestedGreaterGreater",	"&#x02AA2;&#x00338;" },
    { "NotNestedLessLess",	"&#x02AA1;&#x00338;" },
    { "NotRightTriangleBar",	"&#x029D0;&#x00338;" },
    { "NotSquareSubset",	"&#x0228F;&#x00338;" },
    { "NotSquareSuperset",	"&#x02290;&#x00338;" },
    { "NotSucceedsTilde",	"&#x0227F;&#x00338;" },
    { "OverBar",	"&#x000AF;" },
    { "OverBrace",	"&#x0FE37;" },
    { "OverBracket",	"&#x023B4;" },
    { "OverParenthesis",	"&#x0FE35;" },
    { "planckh",	"&#x0210E;" },
    { "Product",	"&#x0220F;" },
    { "rarrb",	"&#x021E5;" },
    { "RightDownTeeVector",	"&#x0295D;" },
    { "RightDownVectorBar",	"&#x02955;" },
    { "RightTeeVector",	"&#x0295B;" },
    { "RightTriangleBar",	"&#x029D0;" },
    { "RightUpDownVector",	"&#x0294F;" },
    { "RightUpTeeVector",	"&#x0295C;" },
    { "RightUpVectorBar",	"&#x02954;" },
    { "RightVectorBar",	"&#x02953;" },
    { "RoundImplies",	"&#x02970;" },
    { "RuleDelayed",	"&#x029F4;" },
    { "Tab",	"&#x00009;" },
    { "ThickSpace",	"&#x02009;&#x0200A;&#x0200A;" },
    { "UnderBar",	"&#x00332;" },
    { "UnderBrace",	"&#x0FE38;" },
    { "UnderBracket",	"&#x023B5;" },
    { "UnderParenthesis",	"&#x0FE36;" },
    { "UpArrowBar",	"&#x02912;" },
    { "Upsilon",	"&#x003A5;" },
    { "VerticalLine",	"&#x0007C;" },
    { "VerticalSeparator",	"&#x02758;" },
    { "ZeroWidthSpace",	"&#x0200B;" },
    { "angle",	"&#x02220;" },
    { "ApplyFunction",	"&#x02061;" },
    { "approx",	"&#x02248;" },
    { "approxeq",	"&#x0224A;" },
    { "Assign",	"&#x02254;" },
    { "backcong",	"&#x0224C;" },
    { "backepsilon",	"&#x003F6;" },
    { "backprime",	"&#x02035;" },
    { "backsim",	"&#x0223D;" },
    { "backsimeq",	"&#x022CD;" },
    { "Backslash",	"&#x02216;" },
    { "barwedge",	"&#x02305;" },
    { "Because",	"&#x02235;" },
    { "because",	"&#x02235;" },
    { "Bernoullis",	"&#x0212C;" },
    { "between",	"&#x0226C;" },
    { "bigcap",	"&#x022C2;" },
    { "bigcirc",	"&#x025EF;" },
    { "bigcup",	"&#x022C3;" },
    { "bigodot",	"&#x02A00;" },
    { "bigoplus",	"&#x02A01;" },
    { "bigotimes",	"&#x02A02;" },
    { "bigsqcup",	"&#x02A06;" },
    { "bigstar",	"&#x02605;" },
    { "bigtriangledown",	"&#x025BD;" },
    { "bigtriangleup",	"&#x025B3;" },
    { "biguplus",	"&#x02A04;" },
    { "bigvee",	"&#x022C1;" },
    { "bigwedge",	"&#x022C0;" },
    { "bkarow",	"&#x0290D;" },
    { "blacklozenge",	"&#x029EB;" },
    { "blacksquare",	"&#x025AA;" },
    { "blacktriangle",	"&#x025B4;" },
    { "blacktriangledown",	"&#x025BE;" },
    { "blacktriangleleft",	"&#x025C2;" },
    { "blacktriangleright",	"&#x025B8;" },
    { "bot",	"&#x022A5;" },
    { "boxminus",	"&#x0229F;" },
    { "boxplus",	"&#x0229E;" },
    { "boxtimes",	"&#x022A0;" },
    { "Breve",	"&#x002D8;" },
    { "bullet",	"&#x02022;" },
    { "Bumpeq",	"&#x0224E;" },
    { "bumpeq",	"&#x0224F;" },
    { "CapitalDifferentialD",	"&#x02145;" },
    { "Cayleys",	"&#x0212D;" },
    { "Cedilla",	"&#x000B8;" },
    { "CenterDot",	"&#x000B7;" },
    { "centerdot",	"&#x000B7;" },
    { "checkmark",	"&#x02713;" },
    { "circeq",	"&#x02257;" },
    { "circlearrowleft",	"&#x021BA;" },
    { "circlearrowright",	"&#x021BB;" },
    { "circledast",	"&#x0229B;" },
    { "circledcirc",	"&#x0229A;" },
    { "circleddash",	"&#x0229D;" },
    { "CircleDot",	"&#x02299;" },
    { "circledR",	"&#x000AE;" },
    { "circledS",	"&#x024C8;" },
    { "CircleMinus",	"&#x02296;" },
    { "CirclePlus",	"&#x02295;" },
    { "CircleTimes",	"&#x02297;" },
    { "ClockwiseContourIntegral",	"&#x02232;" },
    { "CloseCurlyDoubleQuote",	"&#x0201D;" },
    { "CloseCurlyQuote",	"&#x02019;" },
    { "clubsuit",	"&#x02663;" },
    { "coloneq",	"&#x02254;" },
    { "complement",	"&#x02201;" },
    { "complexes",	"&#x02102;" },
    { "Congruent",	"&#x02261;" },
    { "ContourIntegral",	"&#x0222E;" },
    { "Coproduct",	"&#x02210;" },
    { "CounterClockwiseContourIntegral",	"&#x02233;" },
    { "CupCap",	"&#x0224D;" },
    { "curlyeqprec",	"&#x022DE;" },
    { "curlyeqsucc",	"&#x022DF;" },
    { "curlyvee",	"&#x022CE;" },
    { "curlywedge",	"&#x022CF;" },
    { "curvearrowleft",	"&#x021B6;" },
    { "curvearrowright",	"&#x021B7;" },
    { "dbkarow",	"&#x0290F;" },
    { "ddagger",	"&#x02021;" },
    { "ddotseq",	"&#x02A77;" },
    { "Del",	"&#x02207;" },
    { "DiacriticalAcute",	"&#x000B4;" },
    { "DiacriticalDot",	"&#x002D9;" },
    { "DiacriticalDoubleAcute",	"&#x002DD;" },
    { "DiacriticalGrave",	"&#x00060;" },
    { "DiacriticalTilde",	"&#x002DC;" },
    { "Diamond",	"&#x022C4;" },
    { "diamond",	"&#x022C4;" },
    { "diamondsuit",	"&#x02666;" },
    { "DifferentialD",	"&#x02146;" },
    { "digamma",	"&#x003DD;" },
    { "div",	"&#x000F7;" },
    { "divideontimes",	"&#x022C7;" },
    { "doteq",	"&#x02250;" },
    { "doteqdot",	"&#x02251;" },
    { "DotEqual",	"&#x02250;" },
    { "dotminus",	"&#x02238;" },
    { "dotplus",	"&#x02214;" },
    { "dotsquare",	"&#x022A1;" },
    { "doublebarwedge",	"&#x02306;" },
    { "DoubleContourIntegral",	"&#x0222F;" },
    { "DoubleDot",	"&#x000A8;" },
    { "DoubleDownArrow",	"&#x021D3;" },
    { "DoubleLeftArrow",	"&#x021D0;" },
    { "DoubleLeftRightArrow",	"&#x021D4;" },
    { "DoubleLeftTee",	"&#x02AE4;" },
    { "DoubleLongLeftArrow",	"&#x027F8;" },
    { "DoubleLongLeftRightArrow",	"&#x027FA;" },
    { "DoubleLongRightArrow",	"&#x027F9;" },
    { "DoubleRightArrow",	"&#x021D2;" },
    { "DoubleRightTee",	"&#x022A8;" },
    { "DoubleUpArrow",	"&#x021D1;" },
    { "DoubleUpDownArrow",	"&#x021D5;" },
    { "DoubleVerticalBar",	"&#x02225;" },
    { "DownArrow",	"&#x02193;" },
    { "Downarrow",	"&#x021D3;" },
    { "downarrow",	"&#x02193;" },
    { "DownArrowUpArrow",	"&#x021F5;" },
    { "downdownarrows",	"&#x021CA;" },
    { "downharpoonleft",	"&#x021C3;" },
    { "downharpoonright",	"&#x021C2;" },
    { "DownLeftVector",	"&#x021BD;" },
    { "DownRightVector",	"&#x021C1;" },
    { "DownTee",	"&#x022A4;" },
    { "DownTeeArrow",	"&#x021A7;" },
    { "drbkarow",	"&#x02910;" },
    { "Element",	"&#x02208;" },
    { "emptyset",	"&#x02205;" },
    { "eqcirc",	"&#x02256;" },
    { "eqcolon",	"&#x02255;" },
    { "eqsim",	"&#x02242;" },
    { "eqslantgtr",	"&#x02A96;" },
    { "eqslantless",	"&#x02A95;" },
    { "EqualTilde",	"&#x02242;" },
    { "Equilibrium",	"&#x021CC;" },
    { "Exists",	"&#x02203;" },
    { "expectation",	"&#x02130;" },
    { "ExponentialE",	"&#x02147;" },
    { "exponentiale",	"&#x02147;" },
    { "fallingdotseq",	"&#x02252;" },
    { "ForAll",	"&#x02200;" },
    { "Fouriertrf",	"&#x02131;" },
    { "geq",	"&#x02265;" },
    { "geqq",	"&#x02267;" },
    { "geqslant",	"&#x02A7E;" },
    { "gg",	"&#x0226B;" },
    { "ggg",	"&#x022D9;" },
    { "gnapprox",	"&#x02A8A;" },
    { "gneq",	"&#x02A88;" },
    { "gneqq",	"&#x02269;" },
    { "GreaterEqual",	"&#x02265;" },
    { "GreaterEqualLess",	"&#x022DB;" },
    { "GreaterFullEqual",	"&#x02267;" },
    { "GreaterLess",	"&#x02277;" },
    { "GreaterSlantEqual",	"&#x02A7E;" },
    { "GreaterTilde",	"&#x02273;" },
    { "gtrapprox",	"&#x02A86;" },
    { "gtrdot",	"&#x022D7;" },
    { "gtreqless",	"&#x022DB;" },
    { "gtreqqless",	"&#x02A8C;" },
    { "gtrless",	"&#x02277;" },
    { "gtrsim",	"&#x02273;" },
    { "gvertneqq",	"&#x02269;&#x0FE00;" },
    { "Hacek",	"&#x002C7;" },
    { "hbar",	"&#x0210F;" },
    { "heartsuit",	"&#x02665;" },
    { "HilbertSpace",	"&#x0210B;" },
    { "hksearow",	"&#x02925;" },
    { "hkswarow",	"&#x02926;" },
    { "hookleftarrow",	"&#x021A9;" },
    { "hookrightarrow",	"&#x021AA;" },
    { "hslash",	"&#x0210F;" },
    { "HumpDownHump",	"&#x0224E;" },
    { "HumpEqual",	"&#x0224F;" },
    { "iiiint",	"&#x02A0C;" },
    { "iiint",	"&#x0222D;" },
    { "Im",	"&#x02111;" },
    { "ImaginaryI",	"&#x02148;" },
    { "imagline",	"&#x02110;" },
    { "imagpart",	"&#x02111;" },
    { "Implies",	"&#x021D2;" },
    { "in",	"&#x02208;" },
    { "integers",	"&#x02124;" },
    { "Integral",	"&#x0222B;" },
    { "intercal",	"&#x022BA;" },
    { "Intersection",	"&#x022C2;" },
    { "intprod",	"&#x02A3C;" },
    { "InvisibleComma",	"&#x02063;" },
    { "InvisibleTimes",	"&#x02062;" },
    { "langle",	"&#x02329;" },
    { "Laplacetrf",	"&#x02112;" },
    { "lbrace",	"&#x0007B;" },
    { "lbrack",	"&#x0005B;" },
    { "LeftAngleBracket",	"&#x02329;" },
    { "LeftArrow",	"&#x02190;" },
    { "Leftarrow",	"&#x021D0;" },
    { "leftarrow",	"&#x02190;" },
    { "LeftArrowBar",	"&#x021E4;" },
    { "LeftArrowRightArrow",	"&#x021C6;" },
    { "leftarrowtail",	"&#x021A2;" },
    { "LeftCeiling",	"&#x02308;" },
    { "LeftDoubleBracket",	"&#x0301A;" },
    { "LeftDownVector",	"&#x021C3;" },
    { "LeftFloor",	"&#x0230A;" },
    { "leftharpoondown",	"&#x021BD;" },
    { "leftharpoonup",	"&#x021BC;" },
    { "leftleftarrows",	"&#x021C7;" },
    { "LeftRightArrow",	"&#x02194;" },
    { "Leftrightarrow",	"&#x021D4;" },
    { "leftrightarrow",	"&#x02194;" },
    { "leftrightarrows",	"&#x021C6;" },
    { "leftrightharpoons",	"&#x021CB;" },
    { "leftrightsquigarrow",	"&#x021AD;" },
    { "LeftTee",	"&#x022A3;" },
    { "LeftTeeArrow",	"&#x021A4;" },
    { "leftthreetimes",	"&#x022CB;" },
    { "LeftTriangle",	"&#x022B2;" },
    { "LeftTriangleEqual",	"&#x022B4;" },
    { "LeftUpVector",	"&#x021BF;" },
    { "LeftVector",	"&#x021BC;" },
    { "leq",	"&#x02264;" },
    { "leqq",	"&#x02266;" },
    { "leqslant",	"&#x02A7D;" },
    { "lessapprox",	"&#x02A85;" },
    { "lessdot",	"&#x022D6;" },
    { "lesseqgtr",	"&#x022DA;" },
    { "lesseqqgtr",	"&#x02A8B;" },
    { "LessEqualGreater",	"&#x022DA;" },
    { "LessFullEqual",	"&#x02266;" },
    { "LessGreater",	"&#x02276;" },
    { "lessgtr",	"&#x02276;" },
    { "lesssim",	"&#x02272;" },
    { "LessSlantEqual",	"&#x02A7D;" },
    { "LessTilde",	"&#x02272;" },
    { "ll",	"&#x0226A;" },
    { "llcorner",	"&#x0231E;" },
    { "Lleftarrow",	"&#x021DA;" },
    { "lmoustache",	"&#x023B0;" },
    { "lnapprox",	"&#x02A89;" },
    { "lneq",	"&#x02A87;" },
    { "lneqq",	"&#x02268;" },
    { "LongLeftArrow",	"&#x027F5;" },
    { "Longleftarrow",	"&#x027F8;" },
    { "longleftarrow",	"&#x027F5;" },
    { "LongLeftRightArrow",	"&#x027F7;" },
    { "Longleftrightarrow",	"&#x027FA;" },
    { "longleftrightarrow",	"&#x027F7;" },
    { "longmapsto",	"&#x027FC;" },
    { "LongRightArrow",	"&#x027F6;" },
    { "Longrightarrow",	"&#x027F9;" },
    { "longrightarrow",	"&#x027F6;" },
    { "looparrowleft",	"&#x021AB;" },
    { "looparrowright",	"&#x021AC;" },
    { "LowerLeftArrow",	"&#x02199;" },
    { "LowerRightArrow",	"&#x02198;" },
    { "lozenge",	"&#x025CA;" },
    { "lrcorner",	"&#x0231F;" },
    { "Lsh",	"&#x021B0;" },
    { "lvertneqq",	"&#x02268;&#x0FE00;" },
    { "maltese",	"&#x02720;" },
    { "mapsto",	"&#x021A6;" },
    { "measuredangle",	"&#x02221;" },
    { "Mellintrf",	"&#x02133;" },
    { "MinusPlus",	"&#x02213;" },
    { "mp",	"&#x02213;" },
    { "multimap",	"&#x022B8;" },
    { "napprox",	"&#x02249;" },
    { "natural",	"&#x0266E;" },
    { "naturals",	"&#x02115;" },
    { "nearrow",	"&#x02197;" },
    { "NegativeMediumSpace",	"&#x0200B;" },
    { "NegativeThickSpace",	"&#x0200B;" },
    { "NegativeThinSpace",	"&#x0200B;" },
    { "NegativeVeryThinSpace",	"&#x0200B;" },
    { "NestedGreaterGreater",	"&#x0226B;" },
    { "NestedLessLess",	"&#x0226A;" },
    { "nexists",	"&#x02204;" },
    { "ngeq",	"&#x02271;" },
    { "ngeqq",	"&#x02267;&#x00338;" },
    { "ngeqslant",	"&#x02A7E;&#x00338;" },
    { "ngtr",	"&#x0226F;" },
    { "nLeftarrow",	"&#x021CD;" },
    { "nleftarrow",	"&#x0219A;" },
    { "nLeftrightarrow",	"&#x021CE;" },
    { "nleftrightarrow",	"&#x021AE;" },
    { "nleq",	"&#x02270;" },
    { "nleqq",	"&#x02266;&#x00338;" },
    { "nleqslant",	"&#x02A7D;&#x00338;" },
    { "nless",	"&#x0226E;" },
    { "NonBreakingSpace",	"&#x000A0;" },
    { "NotCongruent",	"&#x02262;" },
    { "NotDoubleVerticalBar",	"&#x02226;" },
    { "NotElement",	"&#x02209;" },
    { "NotEqual",	"&#x02260;" },
    { "NotEqualTilde",	"&#x02242;&#x00338;" },
    { "NotExists",	"&#x02204;" },
    { "NotGreater",	"&#x0226F;" },
    { "NotGreaterEqual",	"&#x02271;" },
    { "NotGreaterFullEqual",	"&#x02266;&#x00338;" },
    { "NotGreaterGreater",	"&#x0226B;&#x00338;" },
    { "NotGreaterLess",	"&#x02279;" },
    { "NotGreaterSlantEqual",	"&#x02A7E;&#x00338;" },
    { "NotGreaterTilde",	"&#x02275;" },
    { "NotHumpDownHump",	"&#x0224E;&#x00338;" },
    { "NotLeftTriangle",	"&#x022EA;" },
    { "NotLeftTriangleEqual",	"&#x022EC;" },
    { "NotLess",	"&#x0226E;" },
    { "NotLessEqual",	"&#x02270;" },
    { "NotLessGreater",	"&#x02278;" },
    { "NotLessLess",	"&#x0226A;&#x00338;" },
    { "NotLessSlantEqual",	"&#x02A7D;&#x00338;" },
    { "NotLessTilde",	"&#x02274;" },
    { "NotPrecedes",	"&#x02280;" },
    { "NotPrecedesEqual",	"&#x02AAF;&#x00338;" },
    { "NotPrecedesSlantEqual",	"&#x022E0;" },
    { "NotReverseElement",	"&#x0220C;" },
    { "NotRightTriangle",	"&#x022EB;" },
    { "NotRightTriangleEqual",	"&#x022ED;" },
    { "NotSquareSubsetEqual",	"&#x022E2;" },
    { "NotSquareSupersetEqual",	"&#x022E3;" },
    { "NotSubset",	"&#x02282;&#x020D2;" },
    { "NotSubsetEqual",	"&#x02288;" },
    { "NotSucceeds",	"&#x02281;" },
    { "NotSucceedsEqual",	"&#x02AB0;&#x00338;" },
    { "NotSucceedsSlantEqual",	"&#x022E1;" },
    { "NotSuperset",	"&#x02283;&#x020D2;" },
    { "NotSupersetEqual",	"&#x02289;" },
    { "NotTilde",	"&#x02241;" },
    { "NotTildeEqual",	"&#x02244;" },
    { "NotTildeFullEqual",	"&#x02247;" },
    { "NotTildeTilde",	"&#x02249;" },
    { "NotVerticalBar",	"&#x02224;" },
    { "nparallel",	"&#x02226;" },
    { "nprec",	"&#x02280;" },
    { "npreceq",	"&#x02AAF;&#x00338;" },
    { "nRightarrow",	"&#x021CF;" },
    { "nrightarrow",	"&#x0219B;" },
    { "nshortmid",	"&#x02224;" },
    { "nshortparallel",	"&#x02226;" },
    { "nsimeq",	"&#x02244;" },
    { "nsubset",	"&#x02282;&#x020D2;" },
    { "nsubseteq",	"&#x02288;" },
    { "nsubseteqq",	"&#x02AC5;&#x0338;" },
    { "nsucc",	"&#x02281;" },
    { "nsucceq",	"&#x02AB0;&#x00338;" },
    { "nsupset",	"&#x02283;&#x020D2;" },
    { "nsupseteq",	"&#x02289;" },
    { "nsupseteqq",	"&#x02AC6;&#x0338;" },
    { "ntriangleleft",	"&#x022EA;" },
    { "ntrianglelefteq",	"&#x022EC;" },
    { "ntriangleright",	"&#x022EB;" },
    { "ntrianglerighteq",	"&#x022ED;" },
    { "nwarrow",	"&#x02196;" },
    { "oint",	"&#x0222E;" },
    { "OpenCurlyDoubleQuote",	"&#x0201C;" },
    { "OpenCurlyQuote",	"&#x02018;" },
    { "orderof",	"&#x02134;" },
    { "parallel",	"&#x02225;" },
    { "PartialD",	"&#x02202;" },
    { "pitchfork",	"&#x022D4;" },
    { "PlusMinus",	"&#x000B1;" },
    { "pm",	"&#x000B1;" },
    { "Poincareplane",	"&#x0210C;" },
    { "prec",	"&#x0227A;" },
    { "precapprox",	"&#x02AB7;" },
    { "preccurlyeq",	"&#x0227C;" },
    { "Precedes",	"&#x0227A;" },
    { "PrecedesEqual",	"&#x02AAF;" },
    { "PrecedesSlantEqual",	"&#x0227C;" },
    { "PrecedesTilde",	"&#x0227E;" },
    { "preceq",	"&#x02AAF;" },
    { "precnapprox",	"&#x02AB9;" },
    { "precneqq",	"&#x02AB5;" },
    { "precnsim",	"&#x022E8;" },
    { "precsim",	"&#x0227E;" },
    { "primes",	"&#x02119;" },
    { "Proportion",	"&#x02237;" },
    { "Proportional",	"&#x0221D;" },
    { "propto",	"&#x0221D;" },
    { "quaternions",	"&#x0210D;" },
    { "questeq",	"&#x0225F;" },
    { "rangle",	"&#x0232A;" },
    { "rationals",	"&#x0211A;" },
    { "rbrace",	"&#x0007D;" },
    { "rbrack",	"&#x0005D;" },
    { "Re",	"&#x0211C;" },
    { "realine",	"&#x0211B;" },
    { "realpart",	"&#x0211C;" },
    { "reals",	"&#x0211D;" },
    { "ReverseElement",	"&#x0220B;" },
    { "ReverseEquilibrium",	"&#x021CB;" },
    { "ReverseUpEquilibrium",	"&#x0296F;" },
    { "RightAngleBracket",	"&#x0232A;" },
    { "RightArrow",	"&#x02192;" },
    { "Rightarrow",	"&#x021D2;" },
    { "rightarrow",	"&#x02192;" },
    { "RightArrowBar",	"&#x021E5;" },
    { "RightArrowLeftArrow",	"&#x021C4;" },
    { "rightarrowtail",	"&#x021A3;" },
    { "RightCeiling",	"&#x02309;" },
    { "RightDoubleBracket",	"&#x0301B;" },
    { "RightDownVector",	"&#x021C2;" },
    { "RightFloor",	"&#x0230B;" },
    { "rightharpoondown",	"&#x021C1;" },
    { "rightharpoonup",	"&#x021C0;" },
    { "rightleftarrows",	"&#x021C4;" },
    { "rightleftharpoons",	"&#x021CC;" },
    { "rightrightarrows",	"&#x021C9;" },
    { "rightsquigarrow",	"&#x0219D;" },
    { "RightTee",	"&#x022A2;" },
    { "RightTeeArrow",	"&#x021A6;" },
    { "rightthreetimes",	"&#x022CC;" },
    { "RightTriangle",	"&#x022B3;" },
    { "RightTriangleEqual",	"&#x022B5;" },
    { "RightUpVector",	"&#x021BE;" },
    { "RightVector",	"&#x021C0;" },
    { "risingdotseq",	"&#x02253;" },
    { "rmoustache",	"&#x023B1;" },
    { "Rrightarrow",	"&#x021DB;" },
    { "Rsh",	"&#x021B1;" },
    { "searrow",	"&#x02198;" },
    { "setminus",	"&#x02216;" },
    { "ShortDownArrow",	"&#x02193;" },
    { "ShortLeftArrow",	"&#x02190;" },
    { "shortmid",	"&#x02223;" },
    { "shortparallel",	"&#x02225;" },
    { "ShortRightArrow",	"&#x02192;" },
    { "ShortUpArrow",	"&#x02191;" },
    { "simeq",	"&#x02243;" },
    { "SmallCircle",	"&#x02218;" },
    { "smallsetminus",	"&#x02216;" },
    { "spadesuit",	"&#x02660;" },
    { "Sqrt",	"&#x0221A;" },
    { "sqsubset",	"&#x0228F;" },
    { "sqsubseteq",	"&#x02291;" },
    { "sqsupset",	"&#x02290;" },
    { "sqsupseteq",	"&#x02292;" },
    { "Square",	"&#x025A1;" },
    { "SquareIntersection",	"&#x02293;" },
    { "SquareSubset",	"&#x0228F;" },
    { "SquareSubsetEqual",	"&#x02291;" },
    { "SquareSuperset",	"&#x02290;" },
    { "SquareSupersetEqual",	"&#x02292;" },
    { "SquareUnion",	"&#x02294;" },
    { "Star",	"&#x022C6;" },
    { "straightepsilon",	"&#x003B5;" },
    { "straightphi",	"&#x003D5;" },
    { "Subset",	"&#x022D0;" },
    { "subset",	"&#x02282;" },
    { "subseteq",	"&#x02286;" },
    { "subseteqq",	"&#x02AC5;" },
    { "SubsetEqual",	"&#x02286;" },
    { "subsetneq",	"&#x0228A;" },
    { "subsetneqq",	"&#x02ACB;" },
    { "succ",	"&#x0227B;" },
    { "succapprox",	"&#x02AB8;" },
    { "succcurlyeq",	"&#x0227D;" },
    { "Succeeds",	"&#x0227B;" },
    { "SucceedsEqual",	"&#x02AB0;" },
    { "SucceedsSlantEqual",	"&#x0227D;" },
    { "SucceedsTilde",	"&#x0227F;" },
    { "succeq",	"&#x02AB0;" },
    { "succnapprox",	"&#x02ABA;" },
    { "succneqq",	"&#x02AB6;" },
    { "succnsim",	"&#x022E9;" },
    { "succsim",	"&#x0227F;" },
    { "SuchThat",	"&#x0220B;" },
    { "Sum",	"&#x02211;" },
    { "Superset",	"&#x02283;" },
    { "SupersetEqual",	"&#x02287;" },
    { "Supset",	"&#x022D1;" },
    { "supset",	"&#x02283;" },
    { "supseteq",	"&#x02287;" },
    { "supseteqq",	"&#x02AC6;" },
    { "supsetneq",	"&#x0228B;" },
    { "supsetneqq",	"&#x02ACC;" },
    { "swarrow",	"&#x02199;" },
    { "Therefore",	"&#x02234;" },
    { "therefore",	"&#x02234;" },
    { "thickapprox",	"&#x02248;" },
    { "thicksim",	"&#x0223C;" },
    { "ThinSpace",	"&#x02009;" },
    { "Tilde",	"&#x0223C;" },
    { "TildeEqual",	"&#x02243;" },
    { "TildeFullEqual",	"&#x02245;" },
    { "TildeTilde",	"&#x02248;" },
    { "toea",	"&#x02928;" },
    { "tosa",	"&#x02929;" },
    { "triangle",	"&#x025B5;" },
    { "triangledown",	"&#x025BF;" },
    { "triangleleft",	"&#x025C3;" },
    { "trianglelefteq",	"&#x022B4;" },
    { "triangleq",	"&#x0225C;" },
    { "triangleright",	"&#x025B9;" },
    { "trianglerighteq",	"&#x022B5;" },
    { "TripleDot",	"&#x020DB;" },
    { "twoheadleftarrow",	"&#x0219E;" },
    { "twoheadrightarrow",	"&#x021A0;" },
    { "ulcorner",	"&#x0231C;" },
    { "Union",	"&#x022C3;" },
    { "UnionPlus",	"&#x0228E;" },
    { "UpArrow",	"&#x02191;" },
    { "Uparrow",	"&#x021D1;" },
    { "uparrow",	"&#x02191;" },
    { "UpArrowDownArrow",	"&#x021C5;" },
    { "UpDownArrow",	"&#x02195;" },
    { "Updownarrow",	"&#x021D5;" },
    { "updownarrow",	"&#x02195;" },
    { "UpEquilibrium",	"&#x0296E;" },
    { "upharpoonleft",	"&#x021BF;" },
    { "upharpoonright",	"&#x021BE;" },
    { "UpperLeftArrow",	"&#x02196;" },
    { "UpperRightArrow",	"&#x02197;" },
    { "upsilon",	"&#x003C5;" },
    { "UpTee",	"&#x022A5;" },
    { "UpTeeArrow",	"&#x021A5;" },
    { "upuparrows",	"&#x021C8;" },
    { "urcorner",	"&#x0231D;" },
    { "varepsilon",	"&#x0025B;" },
    { "varkappa",	"&#x003F0;" },
    { "varnothing",	"&#x02205;" },
    { "varphi",	"&#x003C6;" },
    { "varpi",	"&#x003D6;" },
    { "varpropto",	"&#x0221D;" },
    { "varrho",	"&#x003F1;" },
    { "varsigma",	"&#x003C2;" },
    { "varsubsetneq",	"&#x0228A;&#x0FE00;" },
    { "varsubsetneqq",	"&#x02ACB;&#x0FE00;" },
    { "varsupsetneq",	"&#x0228B;&#x0FE00;" },
    { "varsupsetneqq",	"&#x02ACC;&#x0FE00;" },
    { "vartheta",	"&#x003D1;" },
    { "vartriangleleft",	"&#x022B2;" },
    { "vartriangleright",	"&#x022B3;" },
    { "Vee",	"&#x022C1;" },
    { "vee",	"&#x02228;" },
    { "Vert",	"&#x02016;" },
    { "vert",	"&#x0007C;" },
    { "VerticalBar",	"&#x02223;" },
    { "VerticalTilde",	"&#x02240;" },
    { "VeryThinSpace",	"&#x0200A;" },
    { "Wedge",	"&#x022C0;" },
    { "wedge",	"&#x02227;" },
    { "wp",	"&#x02118;" },
    { "wr",	"&#x02240;" },
    { "zeetrf",	"&#x02128;" },
    { 0, 0 }
};

// *******************************************************************
// MmlDocument
// *******************************************************************

QString MmlDocument::fontName(QtMmlWidget::MmlFont type) const
{
    switch (type) {
	case QtMmlWidget::NormalFont:
	    return m_normal_font_name;
	case QtMmlWidget::FrakturFont:
	    return m_fraktur_font_name;
	case QtMmlWidget::SansSerifFont:
	    return m_sans_serif_font_name;
	case QtMmlWidget::ScriptFont:
	    return m_script_font_name;
	case QtMmlWidget::MonospaceFont:
	    return m_monospace_font_name;
	case QtMmlWidget::DoublestruckFont:
	    return m_doublestruck_font_name;
    };

    return QString();
}

void MmlDocument::setFontName(QtMmlWidget::MmlFont type, const QString &name)
{
    switch (type) {
	case QtMmlWidget::NormalFont:
	    m_normal_font_name = name;
	    break;
	case QtMmlWidget::FrakturFont:
	    m_fraktur_font_name = name;
	    break;
	case QtMmlWidget::SansSerifFont:
	    m_sans_serif_font_name = name;
	    break;
	case QtMmlWidget::ScriptFont:
	    m_script_font_name = name;
	    break;
	case QtMmlWidget::MonospaceFont:
	    m_monospace_font_name = name;
	    break;
	case QtMmlWidget::DoublestruckFont:
	    m_doublestruck_font_name = name;
	    break;
    };
}

Mml::NodeType domToMmlNodeType(const QDomNode &dom_node)
{
    Mml::NodeType mml_type = Mml::NoNode;

    switch (dom_node.nodeType()) {
	case QDomNode::ElementNode: {
	    QString tag = dom_node.nodeName();
	    const NodeSpec *spec = mmlFindNodeSpec(tag);

	    // treat urecognised tags as mrow
	    if (spec == 0)
		mml_type = Mml::UnknownNode;
	    else
		mml_type = spec->type;

	    break;
	}
	case QDomNode::TextNode:
	    mml_type = Mml::TextNode;
	    break;

	case QDomNode::DocumentNode:
	    mml_type = Mml::MrowNode;
	    break;

	case QDomNode::EntityReferenceNode:
//	    qWarning("EntityReferenceNode: name=\"" + dom_node.nodeName() + "\" value=\"" + dom_node.nodeValue() + "\"");
	    break;

	case QDomNode::AttributeNode:
	case QDomNode::CDATASectionNode:
	case QDomNode::EntityNode:
	case QDomNode::ProcessingInstructionNode:
	case QDomNode::CommentNode:
	case QDomNode::DocumentTypeNode:
	case QDomNode::DocumentFragmentNode:
	case QDomNode::NotationNode:
	case QDomNode::BaseNode:
	case QDomNode::CharacterDataNode:
	    break;
    }

    return mml_type;
}


MmlDocument::MmlDocument()
{
    m_root_node = 0;

    // Some defaults which happen to work on my computer,
    // but probably won't work on other's
#if defined(Q_WS_WIN)
    m_normal_font_name = "Times New Roman";
#else
    m_normal_font_name = "Century Schoolbook L";
#endif
    m_fraktur_font_name = "Fraktur";
    m_sans_serif_font_name = "Luxi Sans";
    m_script_font_name = "Urw Chancery L";
    m_monospace_font_name = "Luxi Mono";
    m_doublestruck_font_name = "Doublestruck";
    m_base_font_point_size = 16;
    m_foreground_color = Qt::black;
    m_background_color = Qt::white;
}

MmlDocument::~MmlDocument()
{
    clear();
}

void MmlDocument::clear()
{
    delete m_root_node;
    m_root_node = 0;
}

void MmlDocument::dump() const
{
    if (m_root_node == 0)
    	return;

    QString indent;
    _dump(m_root_node, indent);
}

void MmlDocument::_dump(const MmlNode *node, QString &indent) const
{
    if (node == 0) return;

    qWarning("%s", (indent + node->toStr()).toLatin1().data());

    indent += "  ";
    const MmlNode *child = node->firstChild();
    for (; child != 0; child = child->nextSibling())
	_dump(child, indent);
    indent.truncate(indent.length() - 2);
}

bool MmlDocument::setContent(QString text, QString *errorMsg,
				    int *errorLine, int *errorColumn)
{
    clear();

    QString prefix = "<?xml version=\"1.0\"?>\n";
    prefix.append(entityDeclarations());

    uint prefix_lines = 0;
    for (int i = 0; i < prefix.length(); ++i) {
	if (prefix.at(i) == '\n')
	    ++prefix_lines;
    }

    QDomDocument dom;
    if (!dom.setContent(prefix + text, false, errorMsg, errorLine, errorColumn)) {
	if (errorLine != 0)
	    *errorLine -= prefix_lines;
	return false;
    }

    // we don't have access to line info from now on
    if (errorLine != 0) *errorLine = -1;
    if (errorColumn != 0) *errorColumn = -1;

    bool ok;
    MmlNode *root_node = domToMml(dom, &ok, errorMsg);
    if (!ok)
	return false;

    if (root_node == 0) {
	if (errorMsg != 0)
	    *errorMsg = "empty document";
	return false;
    }

    insertChild(0, root_node, 0);
    layout();

/*    QFile of("/tmp/dump.xml");
    of.open(IO_WriteOnly);
    QTextStream os(&of);
    os.setEncoding(QTextStream::UnicodeUTF8);
    os << dom.toString(); */

    return true;
}

void MmlDocument::layout()
{
    if (m_root_node == 0)
    	return;

    m_root_node->layout();
    m_root_node->stretch();
//    dump();
}

bool MmlDocument::insertChild(MmlNode *parent, MmlNode *new_node,
				QString *errorMsg)
{
    if (new_node == 0)
	return true;

    Q_ASSERT(new_node->parent() == 0
		&& new_node->nextSibling() == 0
		&& new_node->previousSibling() == 0);

    if (parent != 0) {
	if (!mmlCheckChildType(parent->nodeType(), new_node->nodeType(), errorMsg))
	    return false;
    }

    if (parent == 0) {
	if (m_root_node == 0)
	    m_root_node = new_node;
	else {
	    MmlNode *n = m_root_node->lastSibling();
	    n->m_next_sibling = new_node;
	    new_node->m_previous_sibling = n;
	}
    }
    else {
	new_node->m_parent = parent;
	if (parent->hasChildNodes()) {
	    MmlNode *n = parent->firstChild()->lastSibling();
	    n->m_next_sibling = new_node;
	    new_node->m_previous_sibling = n;
	}
	else parent->m_first_child = new_node;
    }

    return true;
}

MmlNode *MmlDocument::createNode(NodeType type,
				    const MmlAttributeMap &mml_attr,
				    const QString &mml_value,
				    QString *errorMsg)
{
    Q_ASSERT(type != NoNode);

    MmlNode *mml_node = 0;

    if (!mmlCheckAttributes(type, mml_attr, errorMsg))
	return 0;

    switch (type) {
	case MiNode:
	    mml_node = new MmlMiNode(this, mml_attr);
	    break;
	case MnNode:
	    mml_node = new MmlMnNode(this, mml_attr);
	    break;
	case MfracNode:
	    mml_node = new MmlMfracNode(this, mml_attr);
	    break;
	case MrowNode:
	    mml_node = new MmlMrowNode(this, mml_attr);
	    break;
	case MsqrtNode:
	    mml_node = new MmlMsqrtNode(this, mml_attr);
	    break;
	case MrootNode:
	    mml_node = new MmlMrootNode(this, mml_attr);
	    break;
	case MsupNode:
	    mml_node = new MmlMsupNode(this, mml_attr);
	    break;
	case MsubNode:
	    mml_node = new MmlMsubNode(this, mml_attr);
	    break;
	case MsubsupNode:
	    mml_node = new MmlMsubsupNode(this, mml_attr);
	    break;
	case MoNode:
	    mml_node = new MmlMoNode(this, mml_attr);
	    break;
	case MstyleNode:
	    mml_node = new MmlMstyleNode(this, mml_attr);
	    break;
	case TextNode:
	    mml_node = new MmlTextNode(mml_value, this);
	    break;
	case MphantomNode:
	    mml_node = new MmlMphantomNode(this, mml_attr);
	    break;
	case MfencedNode:
	    mml_node = new MmlMfencedNode(this, mml_attr);
	    break;
	case MtableNode:
	    mml_node = new MmlMtableNode(this, mml_attr);
	    break;
	case MtrNode:
	    mml_node = new MmlMtrNode(this, mml_attr);
	    break;
	case MtdNode:
	    mml_node = new MmlMtdNode(this, mml_attr);
	    break;
	case MoverNode:
	    mml_node = new MmlMoverNode(this, mml_attr);
	    break;
	case MunderNode:
	    mml_node = new MmlMunderNode(this, mml_attr);
	    break;
	case MunderoverNode:
	    mml_node = new MmlMunderoverNode(this, mml_attr);
	    break;
	case MalignMarkNode:
	    mml_node = new MmlMalignMarkNode(this);
	    break;
	case MerrorNode:
	    mml_node = new MmlMerrorNode(this, mml_attr);
	    break;
	case MtextNode:
	    mml_node = new MmlMtextNode(this, mml_attr);
	    break;
	case MpaddedNode:
	    mml_node = new MmlMpaddedNode(this, mml_attr);
	    break;
	case MspaceNode:
	    mml_node = new MmlMspaceNode(this, mml_attr);
	    break;
	case UnknownNode:
	    mml_node = new MmlUnknownNode(this, mml_attr);
	    break;
	case NoNode:
	    mml_node = 0;
	    break;
    }

    return mml_node;
}

void MmlDocument::insertOperator(MmlNode *node, const QString &text)
{
    MmlNode *text_node = createNode(TextNode, MmlAttributeMap(), text, 0);
    MmlNode *mo_node = createNode(MoNode, MmlAttributeMap(), QString(), 0);

    bool ok = insertChild(node, mo_node, 0);
    Q_ASSERT( ok );
    ok = insertChild(mo_node, text_node, 0);
    Q_ASSERT( ok );
}

MmlNode *MmlDocument::domToMml(const QDomNode &dom_node, bool *ok, QString *errorMsg)
{
    // create the node

    Q_ASSERT(ok != 0);

    NodeType mml_type = domToMmlNodeType(dom_node);

    if (mml_type == NoNode) {
	*ok = true;
	return 0;
    }

    QDomNamedNodeMap dom_attr = dom_node.attributes();
    MmlAttributeMap mml_attr;
    for (int i = 0; i < int(dom_attr.length()); ++i) {
	QDomNode attr_node = dom_attr.item(i);
	Q_ASSERT(!attr_node.nodeName().isNull());
	Q_ASSERT(!attr_node.nodeValue().isNull());
	mml_attr[attr_node.nodeName()] = attr_node.nodeValue();
    }

    QString mml_value;
    if (mml_type == TextNode)
	mml_value = dom_node.nodeValue();
    MmlNode *mml_node = createNode(mml_type, mml_attr, mml_value, errorMsg);
    if (mml_node == 0) {
	*ok = false;
	return 0;
    }

    // create the node's children according to the child_spec

    const NodeSpec *spec = mmlFindNodeSpec(mml_type);
    QDomNodeList dom_child_list = dom_node.childNodes();
    int child_cnt = dom_child_list.count();
    MmlNode *mml_child = 0;

    QString separator_list;
    if (mml_type == MfencedNode)
	separator_list = mml_node->explicitAttribute("separators", ",");

    switch (spec->child_spec) {
	case NodeSpec::ChildIgnore:
	    break;

	case NodeSpec::ImplicitMrow:

	    if (child_cnt > 0) {
		mml_child = createImplicitMrowNode(dom_node, ok, errorMsg);
		if (!*ok) {
		    delete mml_node;
		    return 0;
		}

		if (!insertChild(mml_node, mml_child, errorMsg)) {
		    delete mml_node;
		    delete mml_child;
		    *ok = false;
		    return 0;
		}
	    }

	    break;

	default:
	    // exact ammount of children specified - check...
	    if (spec->child_spec != child_cnt) {
		if (errorMsg != 0)
		    *errorMsg = QString("element ")
			+ spec->tag
			+ " requires exactly "
			+ QString::number(spec->child_spec)
			+ " arguments, got "
			+ QString::number(child_cnt);
		delete mml_node;
		*ok = false;
		return 0;
	}

	// ...and continue just as in ChildAny

	case NodeSpec::ChildAny:
	    if (mml_type == MfencedNode)
		insertOperator(mml_node, mml_node->explicitAttribute("open", "("));

	    for (int i = 0; i < child_cnt; ++i) {
		QDomNode dom_child = dom_child_list.item(i);

		MmlNode *mml_child = domToMml(dom_child, ok, errorMsg);
		if (!*ok) {
		    delete mml_node;
		    return 0;
		}

		if (mml_type == MtableNode && mml_child->nodeType() != MtrNode) {
		    MmlNode *mtr_node = createNode(MtrNode, MmlAttributeMap(), QString(), 0);
		    insertChild(mml_node, mtr_node, 0);
		    if (!insertChild(mtr_node, mml_child, errorMsg)) {
			delete mml_node;
			delete mml_child;
			*ok = false;
			return 0;
		    }
		}
		else if (mml_type == MtrNode && mml_child->nodeType() != MtdNode) {
		    MmlNode *mtd_node = createNode(MtdNode, MmlAttributeMap(), QString(), 0);
		    insertChild(mml_node, mtd_node, 0);
		    if (!insertChild(mtd_node, mml_child, errorMsg)) {
			delete mml_node;
			delete mml_child;
			*ok = false;
			return 0;
		    }
		}
		else {
		    if (!insertChild(mml_node, mml_child, errorMsg)) {
			delete mml_node;
			delete mml_child;
			*ok = false;
			return 0;
		    }
		}

		if (i < child_cnt - 1 && mml_type == MfencedNode && !separator_list.isEmpty()) {
		    QChar separator;
		    if (i >= (int)separator_list.length())
			separator = separator_list.at(separator_list.length() - 1);
		    else
			separator = separator_list[i];
		    insertOperator(mml_node, QString(separator));
		}
	    }

	    if (mml_type == MfencedNode)
		insertOperator(mml_node, mml_node->explicitAttribute("close", ")"));

	    break;
    }

    *ok = true;
    return mml_node;
}

MmlNode *MmlDocument::createImplicitMrowNode(const QDomNode &dom_node, bool *ok,
						QString *errorMsg)
{
    QDomNodeList dom_child_list = dom_node.childNodes();
    int child_cnt = dom_child_list.count();

    if (child_cnt == 0) {
	*ok = true;
	return 0;
    }

    if (child_cnt == 1)
	return domToMml(dom_child_list.item(0), ok, errorMsg);

    MmlNode *mml_node = createNode(MrowNode, MmlAttributeMap(),
				    QString(), errorMsg);
    Q_ASSERT(mml_node != 0); // there is no reason in heaven or hell for this to fail

    for (int i = 0; i < child_cnt; ++i) {
	QDomNode dom_child = dom_child_list.item(i);

	MmlNode *mml_child = domToMml(dom_child, ok, errorMsg);
	if (!*ok) {
	    delete mml_node;
	    return 0;
	}

	if (!insertChild(mml_node, mml_child, errorMsg)) {
	    delete mml_node;
	    delete mml_child;
	    *ok = false;
	    return 0;
	}
    }

    return mml_node;
}

void MmlDocument::paint(QPainter *p, const QPoint &pos) const
{
    if (m_root_node == 0)
    	return;

/*    p->save();
    p->setPen(Qt::blue);
    p->drawLine(pos.x() - 5, pos.y(), pos.x() + 5, pos.y());
    p->drawLine(pos.x(), pos.y() - 5, pos.x(), pos.y() + 5);
    p->restore(); */

    QRect mr = m_root_node->myRect();
    m_root_node->setRelOrigin(pos - mr.topLeft());
    m_root_node->paint(p);
}

QSize MmlDocument::size() const
{
    if (m_root_node == 0)
    	return QSize(0, 0);
    return m_root_node->deviceRect().size();
}




// *******************************************************************
// MmlNode
// *******************************************************************


MmlNode::MmlNode(NodeType type, MmlDocument *document, const MmlAttributeMap &attribute_map)
{
    m_parent = 0;
    m_first_child = 0;
    m_next_sibling = 0;
    m_previous_sibling = 0;

    m_node_type = type;
    m_document = document;
    m_attribute_map = attribute_map;

    m_my_rect = m_parent_rect = QRect(0, 0, 0, 0);
    m_rel_origin = QPoint(0, 0);
    m_stretched = false;
}

MmlNode::~MmlNode()
{
    MmlNode *n = firstChild();
    while (n != 0) {
	MmlNode *tmp = n->nextSibling();
	delete n;
	n = tmp;
    }
}

static QString rectToStr(const QRect &rect)
{
    return QString("[(%1, %2), %3x%4]")
		.arg(rect.x())
		.arg(rect.y())
		.arg(rect.width())
		.arg(rect.height());
}

QString MmlNode::toStr() const
{
    const NodeSpec *spec = mmlFindNodeSpec(nodeType());
    Q_ASSERT(spec != 0);

    return QString("%1 %2 mr=%3 pr=%4 dr=%5 ro=(%7, %8) str=%9")
		.arg(spec->type_str)
		.arg((unsigned long)this, 0, 16)
		.arg(rectToStr(myRect()))
		.arg(rectToStr(parentRect()))
		.arg(rectToStr(deviceRect()))
		.arg(m_rel_origin.x())
		.arg(m_rel_origin.y())
		.arg((int)isStretched());
}

int MmlNode::interpretSpacing(const QString &value, bool *ok) const
{
    return ::interpretSpacing(value, em(), ex(), ok);
}

int MmlNode::basePos() const
{
    QFontMetrics fm(font());
    return fm.strikeOutPos();
}

int MmlNode::underlinePos() const
{
    QFontMetrics fm(font());
    return basePos() + fm.underlinePos();
}
int MmlNode::overlinePos() const
{
    QFontMetrics fm(font());
    return basePos() - fm.overlinePos();
}

MmlNode *MmlNode::lastSibling() const
{
    const MmlNode *n = this;
    while (!n->isLastSibling())
	n = n->nextSibling();
    return const_cast<MmlNode*>(n);
}

MmlNode *MmlNode::firstSibling() const
{
    const MmlNode *n = this;
    while (!n->isFirstSibling())
	n = n->previousSibling();
    return const_cast<MmlNode*>(n);
}

int MmlNode::em() const
{
    return QFontMetrics(font()).boundingRect('m').width();
}

int MmlNode::ex() const
{
    return QFontMetrics(font()).boundingRect('x').height();
}

int MmlNode::scriptlevel(const MmlNode *) const
{
    int parent_sl;
    const MmlNode *p = parent();
    if (p == 0)
	parent_sl = 0;
    else
	parent_sl = p->scriptlevel(this);

    QString expl_sl_str = explicitAttribute("scriptlevel");
    if (expl_sl_str.isNull())
	return parent_sl;

    if (expl_sl_str.startsWith("+") || expl_sl_str.startsWith("-")) {
	bool ok;
	int expl_sl = expl_sl_str.toInt(&ok);
	if (ok) {
	    return parent_sl + expl_sl;
	}
	else {
	    qWarning("MmlNode::scriptlevel(): bad value %s", expl_sl_str.toLatin1().data());
	    return parent_sl;
	}
    }

    bool ok;
    int expl_sl = expl_sl_str.toInt(&ok);
    if (ok)
	return expl_sl;


    if (expl_sl_str == "+")
	return parent_sl + 1;
    else if (expl_sl_str == "-")
	return parent_sl - 1;
    else {
        qWarning("MmlNode::scriptlevel(): could not parse value: \"%s\"", expl_sl_str.toLatin1().data());
	return parent_sl;
    }
}

QPoint MmlNode::devicePoint(const QPoint &p) const
{
    QRect mr = myRect();
    QRect dr = deviceRect();

    if (isStretched())
	return dr.topLeft() + QPoint((p.x() - mr.left())*dr.width()/mr.width(),
					(p.y() - mr.top())*dr.height()/mr.height());
    else
	return dr.topLeft() + p - mr.topLeft();
}

QString MmlNode::inheritAttributeFromMrow(const QString &name,
					    const QString &def) const
{
    const MmlNode *p = this;
    for (; p != 0; p = p->parent()) {
	if (p == this || p->nodeType() == MstyleNode) {
	    QString value = p->explicitAttribute(name);
	    if (!value.isNull())
		return value;
	}
    }

    return def;
}

QColor MmlNode::color() const
{
    // If we are child of <merror> return red
    const MmlNode *p = this;
    for (; p != 0; p = p->parent()) {
	if (p->nodeType() == MerrorNode)
	    return QColor("red");
    }

    QString value_str = inheritAttributeFromMrow("mathcolor");
    if (value_str.isNull())
	value_str = inheritAttributeFromMrow("color");
    if (value_str.isNull())
	return QColor();

    return QColor(value_str);
}

QColor MmlNode::background() const
{
    QString value_str = inheritAttributeFromMrow("mathbackground");
    if (value_str.isNull())
	value_str = inheritAttributeFromMrow("background");
    if (value_str.isNull())
	return QColor();

    return QColor(value_str);
}

static void updateFontAttr(MmlAttributeMap &font_attr, const MmlNode *n,
				const QString &name, const QString &preferred_name = QString())
{
    if (font_attr.contains(preferred_name) || font_attr.contains(name))
	return;
    QString value = n->explicitAttribute(name);
    if (!value.isNull())
	font_attr[name] = value;
}

static MmlAttributeMap collectFontAttributes(const MmlNode *node)
{
    MmlAttributeMap font_attr;

    for (const MmlNode *n = node; n != 0; n = n->parent()) {
	if (n == node || n->nodeType() == Mml::MstyleNode) {
	    updateFontAttr(font_attr, n, "mathvariant");
	    updateFontAttr(font_attr, n, "mathsize");

	    // depreciated attributes
	    updateFontAttr(font_attr, n, "fontsize", "mathsize");
	    updateFontAttr(font_attr, n, "fontweight", "mathvariant");
	    updateFontAttr(font_attr, n, "fontstyle", "mathvariant");
	    updateFontAttr(font_attr, n, "fontfamily", "mathvariant");
	}
    }

    return font_attr;
}

QFont MmlNode::font() const
{
    QFont fn(document()->fontName(QtMmlWidget::NormalFont),
		document()->baseFontPointSize());

    int ps = fn.pointSize();
    int sl = scriptlevel();
    if (sl >= 0) {
	for (int i = 0; i < sl; ++i)
	    ps = (int)(ps*g_script_size_multiplier);
    }
    else {
	for (int i = 0; i > sl; --i)
	    ps = (int)(ps/g_script_size_multiplier);
    }
    if (ps < g_min_font_point_size)
	ps = g_min_font_point_size;
    fn.setPointSize(ps);

    int em = QFontMetrics(fn).boundingRect('m').width();
    int ex = QFontMetrics(fn).boundingRect('x').height();

    MmlAttributeMap font_attr = collectFontAttributes(this);

    if (font_attr.contains("mathvariant")) {
	QString value = font_attr["mathvariant"];

	bool ok;
	uint mv = interpretMathVariant(value, &ok);

	if (ok) {
	    if (mv & ScriptMV)
		fn.setFamily(document()->fontName(QtMmlWidget::ScriptFont));

	    if (mv & FrakturMV)
		fn.setFamily(document()->fontName(QtMmlWidget::FrakturFont));

	    if (mv & SansSerifMV)
		fn.setFamily(document()->fontName(QtMmlWidget::SansSerifFont));

	    if (mv & MonospaceMV)
		fn.setFamily(document()->fontName(QtMmlWidget::MonospaceFont));

	    if (mv & DoubleStruckMV)
		fn.setFamily(document()->fontName(QtMmlWidget::DoublestruckFont));

	    if (mv & BoldMV)
		fn.setBold(true);

	    if (mv & ItalicMV)
		fn.setItalic(true);
	}
    }

    if (font_attr.contains("mathsize")) {
	QString value = font_attr["mathsize"];
	fn = interpretMathSize(value, fn, em, ex, 0);
    }

    fn = interpretDepreciatedFontAttr(font_attr, fn, em, ex);

    if (nodeType() == MiNode
	    && !font_attr.contains("mathvariant")
	    && !font_attr.contains("fontstyle")) {
	const MmlMiNode *mi_node = (const MmlMiNode*) this;
	if (mi_node->text().length() == 1)
	    fn.setItalic(true);
    }

    if (nodeType() == MoNode) {
	fn.setItalic(false);
	fn.setBold(false);
    }

    return fn;
}

QString MmlNode::explicitAttribute(const QString &name, const QString &def) const
{
    MmlAttributeMap::const_iterator it = m_attribute_map.find(name);
    if (it != m_attribute_map.end())
	return *it;
    return def;
}


QRect MmlNode::parentRect() const
{
    if (isStretched())
	return m_parent_rect;

    QRect mr = myRect();
    QPoint ro = relOrigin();

    return QRect(ro + mr.topLeft(), mr.size());
}


void MmlNode::stretchTo(const QRect &rect)
{
    m_parent_rect = rect;
    m_stretched = true;
}

void MmlNode::setRelOrigin(const QPoint &rel_origin)
{
    m_rel_origin = rel_origin + QPoint(-myRect().left(), 0);
    m_stretched = false;
}

void MmlNode::updateMyRect()
{
    m_my_rect = symbolRect();
    MmlNode *child = firstChild();
    for (; child != 0; child = child->nextSibling())
	m_my_rect |= child->parentRect();
}

void MmlNode::layout()
{
    m_parent_rect = QRect(0, 0, 0, 0);
    m_stretched = false;
    m_rel_origin = QPoint(0, 0);

    MmlNode *child = firstChild();
    for (; child != 0; child = child->nextSibling())
	child->layout();

    layoutSymbol();

    updateMyRect();

    if (parent() == 0)
	m_rel_origin = QPoint(0, 0);
}


QRect MmlNode::deviceRect() const
{
    if (parent() == 0)
	return QRect(relOrigin() + myRect().topLeft(), myRect().size());

/*    if (!isStretched()) {
	QRect pdr = parent()->deviceRect();
	QRect pmr = parent()->myRect();
	QRect pr = parentRect();
	QRect mr = myRect();
	return QRect(pdr.left() + pr.left() - pmr.left(),
			pdr.top()  + pr.top() - pmr.top(),
			mr.width(), mr.height());
    }
*/
    QRect pdr = parent()->deviceRect();
    QRect pr = parentRect();
    QRect pmr = parent()->myRect();

    float scale_w = 0;
    if (pmr.width() != 0)
	scale_w = (float)pdr.width()/pmr.width();
    float scale_h = 0;
    if (pmr.height() != 0)
	scale_h = (float)pdr.height()/pmr.height();

    return QRect(pdr.left() + ROUND((pr.left() - pmr.left())*scale_w),
		     pdr.top()  + ROUND((pr.top() - pmr.top())*scale_h),
		     ROUND((pr.width()*scale_w)),
		     ROUND((pr.height()*scale_h)));
}

void MmlNode::layoutSymbol()
{
    // default behaves like an mrow

    // now lay them out in a neat row, aligning their origins to my origin
    int w = 0;
    MmlNode *child = firstChild();
    for (; child != 0; child = child->nextSibling()) {
	child->setRelOrigin(QPoint(w, 0));
	w += child->parentRect().width() + 1;
    }
}

void MmlNode::paint(QPainter *p)
{
    if (!myRect().isValid())
        return;
    p->save();
    p->setViewport(deviceRect());
    p->setWindow(myRect());


    QColor fg = color();
    QColor bg = background();
    if (bg.isValid())
	p->fillRect(myRect(), bg);
    if (fg.isValid())
	p->setPen(color());

    MmlNode *child = firstChild();
    for (; child != 0; child = child->nextSibling())
	child->paint(p);

    paintSymbol(p);

    p->restore();
}

void MmlNode::paintSymbol(QPainter *p) const
{
    if (g_draw_frames && myRect().isValid()) {
	p->save();
	p->setPen(Qt::red);
	p->drawRect(m_my_rect);
	QPen pen = p->pen();
	pen.setStyle(Qt::DotLine);
	p->setPen(pen);
	p->drawLine(myRect().left(), 0, myRect().right(), 0);
	p->restore();
    }
}

void MmlNode::stretch()
{
    MmlNode *child = firstChild();
    for (; child != 0; child = child->nextSibling())
	child->stretch();
}

QString MmlTokenNode::text() const
{
    QString result;

    const MmlNode *child = firstChild();
    for (; child != 0; child = child->nextSibling()) {
	if (child->nodeType() != TextNode) continue;
	if (!result.isEmpty())
	    result += ' ';
	result += ((MmlTextNode*)child)->text();
    }

    return result;
}

MmlNode *MmlMfracNode::numerator() const
{
    MmlNode *node = firstChild();
    Q_ASSERT(node != 0);
    return node;
}

MmlNode *MmlMfracNode::denominator() const
{
    MmlNode *node = numerator()->nextSibling();
    Q_ASSERT(node != 0);
    return node;
}

QRect MmlMfracNode::symbolRect() const
{
    int num_width = numerator()->myRect().width();
    int denom_width = denominator()->myRect().width();
    int my_width = qMax(num_width, denom_width) + 4;

    return QRect(-my_width/2, 0, my_width, 1);
}

void MmlMfracNode::layoutSymbol()
{
    MmlNode *num = numerator();
    MmlNode *denom = denominator();

    QRect num_rect = num->myRect();
    QRect denom_rect = denom->myRect();

    int spacing = (int)(g_mfrac_spacing*(num_rect.height() + denom_rect.height()));

    num->setRelOrigin(QPoint(-num_rect.width()/2, - spacing - num_rect.bottom()));
    denom->setRelOrigin(QPoint(-denom_rect.width()/2, spacing - denom_rect.top()));
}

static bool zeroLineThickness(const QString &s)
{
    if (s.length() == 0 || !s[0].isDigit())
	return false;

    for (int i = 0; i < s.length(); ++i) {
	QChar c = s.at(i);
	if (c.isDigit() && c != '0')
	    return false;
    }
    return true;
}

void MmlMfracNode::paintSymbol(QPainter *p) const
{
    QString linethickness_str = inheritAttributeFromMrow("linethickness", "1");

    /* InterpretSpacing returns an int, which might be 0 even if the thickness
       is > 0, though very very small. That's ok, because the painter then paints
       a line of thickness 1. However, we have to run this check if the line
       thickness really is zero */
    if (!zeroLineThickness(linethickness_str)) {
	bool ok;
	int linethickness = interpretSpacing(linethickness_str, &ok);
	if (!ok)
	    linethickness = 1;

	p->save();
	QPen pen = p->pen();
	pen.setWidth(linethickness);
	p->setPen(pen);
	QSize s = myRect().size();
	p->drawLine(-s.width()/2, 0, s.width()/2, 0);
	p->restore();
    }
}

MmlNode *MmlRootBaseNode::base() const
{
    MmlNode *node = firstChild();
//    Q_ASSERT(node != 0);
    return node;
}

MmlNode *MmlRootBaseNode::index() const
{
    MmlNode *b = base();
    if (b == 0)
	return 0;
    return b->nextSibling();
}

int MmlRootBaseNode::scriptlevel(const MmlNode *child) const
{
    int sl = MmlNode::scriptlevel();

    MmlNode *i = index();
    if (child != 0 && child == i)
	return sl + 1;
    else
	return sl;
}


QRect MmlRootBaseNode::symbolRect() const
{
    MmlNode *b = base();
    QRect base_rect;
    if (b == 0)
	base_rect = QRect(0, 0, 1, 1);
    else
	base_rect = base()->myRect();

    int margin = (int)(g_mroot_base_margin*base_rect.height());
    int tw = tailWidth();

    return QRect(-tw, base_rect.top() - margin, tw,
		    base_rect.height() + 2*margin);
}

int MmlRootBaseNode::tailWidth() const
{
    QFontMetrics fm(font());
    return fm.boundingRect(g_radical_char).width();
}

void MmlRootBaseNode::layoutSymbol()
{
    MmlNode *b = base();
    QSize base_size;
    if (b != 0) {
	b->setRelOrigin(QPoint(0, 0));
	base_size = base()->myRect().size();
    } else
	base_size = QSize(1, 1);

    MmlNode *i = index();
    if (i != 0) {
	int tw = tailWidth();

	QRect i_rect = i->myRect();
	i->setRelOrigin(QPoint(-tw/2 - i_rect.width(),
				-i_rect.bottom() - 4));
    }
}

void MmlRootBaseNode::paintSymbol(QPainter *p) const
{
    QFont fn = font();

    p->save();

    QRect sr = symbolRect();

    QRect r = sr;
    r.moveTopLeft(devicePoint(sr.topLeft()));
    p->setViewport(r);
    p->setWindow(QFontMetrics(fn).boundingRect(g_radical_char));
    p->setFont(font());
    p->drawText(0, 0, QString(g_radical_char));

    p->restore();

    p->drawLine(sr.right(), sr.top(), myRect().right(), sr.top());
}

MmlTextNode::MmlTextNode(const QString &text, MmlDocument *document)
    : MmlNode(TextNode, document, MmlAttributeMap())
{
    m_text = text;
    // Trim whitespace from ends, but keep nbsp and thinsp
    static const QRegularExpression leadingWhitespace("^[^\\S\\x00a0\\x2009]+");
    static const QRegularExpression trailingWhitespace("[^\\S\\x00a0\\x2009]+$");
    m_text.remove(leadingWhitespace);
    m_text.remove(trailingWhitespace);

    if (m_text == QString(QChar(0x62, 0x20))	     // &InvisibleTimes;
	    || m_text == QString(QChar(0x63, 0x20))  // &InvisibleComma;
	    || m_text == QString(QChar(0x61, 0x20))) // &ApplyFunction;
	m_text = "";
}

QString MmlTextNode::toStr() const
{
    return MmlNode::toStr() + ", text=\"" + m_text + "\"";
}

void MmlTextNode::paintSymbol(QPainter *p) const
{
    MmlNode::paintSymbol(p);

    QFont fn = font();

    QFontInfo fi(fn);
//    qWarning("MmlTextNode::paintSymbol(): requested: %s, used: %s, size=%d, italic=%d, bold=%d, text=\"%s\" sl=%d",
//    	    	fn.family().latin1(), fi.family().latin1(), fi.pointSize(), (int)fi.italic(), (int)fi.bold(), m_text.latin1(), scriptlevel());

    QFontMetrics fm(fn);

    p->save();
    p->setFont(fn);

    p->drawText(0, fm.strikeOutPos(), m_text);
    p->restore();
}

QRect MmlTextNode::symbolRect() const
{
    QFontMetrics fm(font());

    QRect br = fm.tightBoundingRect(m_text);
    br.translate(0, fm.strikeOutPos());

    return br;
}

MmlNode *MmlSubsupBaseNode::base() const
{
    MmlNode *b = firstChild();
    Q_ASSERT(b != 0);
    return b;
}

MmlNode *MmlSubsupBaseNode::sscript() const
{
    MmlNode *s = base()->nextSibling();
    Q_ASSERT(s != 0);
    return s;
}

int MmlSubsupBaseNode::scriptlevel(const MmlNode *child) const
{
    int sl = MmlNode::scriptlevel();

    MmlNode *s = sscript();
    if (child != 0 && child == s)
	return sl + 1;
    else
	return sl;
}

void MmlMsupNode::layoutSymbol()
{
    MmlNode *b = base();
    MmlNode *s = sscript();

    b->setRelOrigin(QPoint(-b->myRect().width(), 0));
    s->setRelOrigin(QPoint(0, b->myRect().top()));
}

void MmlMsubNode::layoutSymbol()
{
    MmlNode *b = base();
    MmlNode *s = sscript();

    b->setRelOrigin(QPoint(-b->myRect().width(), 0));
    s->setRelOrigin(QPoint(0, b->myRect().bottom()));
}

MmlNode *MmlMsubsupNode::base() const
{
    MmlNode *b = firstChild();
    Q_ASSERT(b != 0);
    return b;
}

MmlNode *MmlMsubsupNode::subscript() const
{
    MmlNode *sub = base()->nextSibling();
    Q_ASSERT(sub != 0);
    return sub;
}

MmlNode *MmlMsubsupNode::superscript() const
{
    MmlNode *sup = subscript()->nextSibling();
    Q_ASSERT(sup != 0);
    return sup;
}

void MmlMsubsupNode::layoutSymbol()
{
    MmlNode *b = base();
    MmlNode *sub = subscript();
    MmlNode *sup = superscript();

    b->setRelOrigin(QPoint(-b->myRect().width(), 0));
    sub->setRelOrigin(QPoint(0, b->myRect().bottom()));
    sup->setRelOrigin(QPoint(0, b->myRect().top()));
}

int MmlMsubsupNode::scriptlevel(const MmlNode *child) const
{
    int sl = MmlNode::scriptlevel();

    MmlNode *sub = subscript();
    MmlNode *sup = superscript();

    if (child != 0 && (child == sup || child == sub))
	return sl + 1;
    else
	return sl;
}

QString MmlMoNode::toStr() const
{
    return MmlNode::toStr() + QString(" form=%1").arg((int)form());
}

void MmlMoNode::layoutSymbol()
{
    MmlNode *child = firstChild();
    if (child == 0)
	return;

    child->setRelOrigin(QPoint(0, 0));

    if (m_oper_spec == 0)
	m_oper_spec = mmlFindOperSpec(text(), form());
}

MmlMoNode::MmlMoNode(MmlDocument *document, const MmlAttributeMap &attribute_map)
    : MmlTokenNode(MoNode, document, attribute_map)
{
    m_oper_spec = 0;
}

QString MmlMoNode::dictionaryAttribute(const QString &name) const
{
    const MmlNode *p = this;
    for (; p != 0; p = p->parent()) {
	if (p == this || p->nodeType() == MstyleNode) {
	    QString expl_attr = p->explicitAttribute(name);
	    if (!expl_attr.isNull())
		return expl_attr;
	}
    }

    return mmlDictAttribute(name, m_oper_spec);
}

Mml::FormType MmlMoNode::form() const
{
    QString value_str = inheritAttributeFromMrow("form");
    if (!value_str.isNull()) {
	bool ok;
	FormType value = interpretForm(value_str, &ok);
	if (ok)
	    return value;
	else
	    qWarning("Could not convert %s to form", value_str.toLatin1().data());

    }

    // Default heuristic.
    if (firstSibling() == (MmlNode*)this && lastSibling() != (MmlNode*)this)
	return PrefixForm;
    else if (lastSibling() == (MmlNode*)this && firstSibling() != (MmlNode*)this)
	return PostfixForm;
    else return InfixForm;

}

void MmlMoNode::stretch()
{
    if (parent() == 0)
	return;

    if (m_oper_spec == 0)
	return;

    if (m_oper_spec->stretch_dir == OperSpec::HStretch
    	    && parent()->nodeType() == MrowNode
	    && (nextSibling() != 0 || previousSibling() != 0))
	return;

    QRect pmr = parent()->myRect();
    QRect pr = parentRect();

    switch (m_oper_spec->stretch_dir) {
	case OperSpec::VStretch:
	    stretchTo(QRect(pr.left(), pmr.top(), pr.width(), pmr.height()));
	    break;
	case OperSpec::HStretch:
	    stretchTo(QRect(pmr.left(), pr.top(), pmr.width(), pr.height()));
	    break;
	case OperSpec::HVStretch:
	    stretchTo(pmr);
	    break;
	case OperSpec::NoStretch:
	    break;
    }
}

int MmlMoNode::lspace() const
{
    Q_ASSERT(m_oper_spec != 0);
    if (parent() == 0
	    || (parent()->nodeType() != MrowNode
	    	    && parent()->nodeType() != MfencedNode
		    && parent()->nodeType() != UnknownNode)
	    || (previousSibling() == 0 && nextSibling() == 0))
	return 0;
    else
	return interpretSpacing(dictionaryAttribute("lspace"), 0);
}

int MmlMoNode::rspace() const
{
    Q_ASSERT(m_oper_spec != 0);
    if (parent() == 0
	    || (parent()->nodeType() != MrowNode
	    	    && parent()->nodeType() != MfencedNode
		    && parent()->nodeType() != UnknownNode)
	    || (previousSibling() == 0 && nextSibling() == 0))
	return 0;
    else
	return interpretSpacing(dictionaryAttribute("rspace"), 0);
}

QRect MmlMoNode::symbolRect() const
{
    const MmlNode *child = firstChild();

    if (child == 0)
	return QRect(0, 0, 0, 0);

    QRect cmr = child->myRect();

    return QRect(-lspace(), cmr.top(),
		   cmr.width() + lspace() + rspace(), cmr.height());
}

int MmlMtableNode::rowspacing() const
{
    QString value = explicitAttribute("rowspacing");
    if (value.isNull())
	return ex();
    bool ok;
    int r = interpretSpacing(value, &ok);

    if (ok)
	return r;
    else
	return ex();
}

int MmlMtableNode::columnspacing() const
{
    QString value = explicitAttribute("columnspacing");
    if (value.isNull())
	return (int)(0.8*em());
    bool ok;
    int r = interpretSpacing(value, &ok);

    if (ok)
	return r;
    else
	return (int)(0.8*em());
}

uint MmlMtableNode::CellSizeData::colWidthSum() const
{
    uint w = 0;
    for (int i = 0; i < col_widths.count(); ++i)
	w += col_widths[i];
    return w;
}

uint MmlMtableNode::CellSizeData::rowHeightSum() const
{
    uint h = 0;
    for (int i = 0; i < row_heights.count(); ++i)
	h += row_heights[i];
    return h;
}

void MmlMtableNode::CellSizeData::init(const MmlNode *first_row)
{
    col_widths.clear();
    row_heights.clear();

    const MmlNode *mtr = first_row;
    for (; mtr != 0; mtr = mtr->nextSibling()) {

	Q_ASSERT(mtr->nodeType() == MtrNode);

	int col_cnt = 0;
	const MmlNode *mtd = mtr->firstChild();
	for (; mtd != 0; mtd = mtd->nextSibling(), ++col_cnt) {

	    Q_ASSERT(mtd->nodeType() == MtdNode);

	    QRect mtdmr = mtd->myRect();

	    if (col_cnt == col_widths.count())
		col_widths.append(mtdmr.width());
	    else
		col_widths[col_cnt] = qMax(col_widths[col_cnt], mtdmr.width());
	}

	row_heights.append(mtr->myRect().height());
    }
}

void MmlMtableNode::layoutSymbol()
{
    // Obtain natural widths of columns
    m_cell_size_data.init(firstChild());

    int col_spc = columnspacing();
    int row_spc = rowspacing();
    int frame_spc_hor = framespacing_hor();
    QString columnwidth_attr = explicitAttribute("columnwidth", "auto");

    // Is table width set by user? If so, set col_width_sum and never ever change it.
    int col_width_sum = m_cell_size_data.colWidthSum();
    bool width_set_by_user = false;
    QString width_str = explicitAttribute("width", "auto");
    if (width_str != "auto") {
	bool ok;

	int w = interpretSpacing(width_str, &ok);
	if (ok) {
	    col_width_sum = w
			    - col_spc*(m_cell_size_data.numCols() - 1)
			    - frame_spc_hor*2;
	    width_set_by_user = true;
	}
    }

    // Find out what kind of columns we are dealing with and set the widths of
    // statically sized columns.
    int fixed_width_sum = 0;	      // sum of widths of statically sized set columns
    int auto_width_sum = 0; 	      // sum of natural widths of auto sized columns
    int relative_width_sum = 0;       // sum of natural widths of relatively sized columns
    double relative_fraction_sum = 0; // total fraction of width taken by relatively
				      // sized columns
    int i;
    for (i = 0; i < m_cell_size_data.numCols(); ++i) {
	QString value = interpretListAttr(columnwidth_attr, i, "auto");

	// Is it an auto sized column?
	if (value == "auto" || value == "fit") {
	    auto_width_sum += m_cell_size_data.col_widths[i];
	    continue;
	}

	// Is it a statically sized column?
	bool ok;
	int w = interpretSpacing(value, &ok);
	if (ok) {
	    // Yup, sets its width to the user specified value
	    m_cell_size_data.col_widths[i] = w;
	    fixed_width_sum += w;
	    continue;
	}

	// Is it a relatively sized column?
	if (value.endsWith("%")) {
	    value.truncate(value.length() - 1);
	    double factor = value.toFloat(&ok);
	    if (ok && !value.isEmpty()) {
		factor /= 100.0;
		relative_width_sum += m_cell_size_data.col_widths[i];
		relative_fraction_sum += factor;
		if (!width_set_by_user) {
		    // If the table width was not set by the user, we are free to increase
		    // it so that the width of this column will be >= than its natural width
		    int min_col_width_sum = ROUND(m_cell_size_data.col_widths[i]/factor);
		    if (min_col_width_sum > col_width_sum)
			col_width_sum = min_col_width_sum;
		}
		continue;
	    }
	    else
		qWarning("MmlMtableNode::layoutSymbol(): could not parse value %s%%", value.toLatin1().data());
	}

	// Relatively sized column, but we failed to parse the factor. Treat is like an auto
	// column.
	auto_width_sum += m_cell_size_data.col_widths[i];
    }

    // Work out how much space remains for the auto olumns, after allocating
    // the statically sized and the relatively sized columns.
    int required_auto_width_sum = col_width_sum
				    - ROUND(relative_fraction_sum*col_width_sum)
				    - fixed_width_sum;

    if (!width_set_by_user && required_auto_width_sum < auto_width_sum) {
	if (relative_fraction_sum < 1)
	    col_width_sum = ROUND((fixed_width_sum + auto_width_sum)/(1 - relative_fraction_sum));
	else
	    col_width_sum = fixed_width_sum + auto_width_sum + relative_width_sum;
	required_auto_width_sum = auto_width_sum;
    }

    // Ratio by which we have to shring/grow all auto sized columns to make it all fit
    double auto_width_scale = 1;
    if (auto_width_sum > 0)
	auto_width_scale = (float)required_auto_width_sum/auto_width_sum;

    // Set correct sizes for the auto sized and the relatively sized columns.
    for (i = 0; i < m_cell_size_data.numCols(); ++i) {
	QString value = interpretListAttr(columnwidth_attr, i, "auto");

	// Is it a relatively sized column?
	if (value.endsWith("%")) {
	    bool ok;
	    int w = interpretPercentSpacing(value, col_width_sum, &ok);
	    if (ok)
		m_cell_size_data.col_widths[i] = w;
	    else
		// We're treating parsing errors here as auto sized columns
		m_cell_size_data.col_widths[i]
			= ROUND(auto_width_scale*m_cell_size_data.col_widths[i]);
	}
	// Is it an auto sized column?
	else if (value == "auto") {
	    m_cell_size_data.col_widths[i]
		    = ROUND(auto_width_scale*m_cell_size_data.col_widths[i]);
	}
    }

    QString s;
    QList<int> &col_widths = m_cell_size_data.col_widths;
    for (i = 0; i < col_widths.count(); ++i) {
	s += QString("[w=%1 %2%%]")
		.arg(col_widths[i])
		.arg(100*col_widths[i]/m_cell_size_data.colWidthSum());
    }
//    qWarning(s);

    m_content_width = m_cell_size_data.colWidthSum()
		    + col_spc*(m_cell_size_data.numCols() - 1);
    m_content_height = m_cell_size_data.rowHeightSum()
		    + row_spc*(m_cell_size_data.numRows() - 1);

    int bottom = -m_content_height/2;
    MmlNode *child = firstChild();
    for (; child != 0; child = child->nextSibling()) {
	Q_ASSERT(child->nodeType() == MtrNode);
	MmlMtrNode *row = (MmlMtrNode*) child;

	row->layoutCells(m_cell_size_data.col_widths, col_spc);
	QRect rmr = row->myRect();
	row->setRelOrigin(QPoint(0, bottom - rmr.top()));
	bottom += rmr.height() + row_spc;
    }
}

QRect MmlMtableNode::symbolRect() const
{
    int frame_spc_hor = framespacing_hor();
    int frame_spc_ver = framespacing_ver();

    return QRect(-frame_spc_hor,
		    -m_content_height/2 - frame_spc_ver,
		    m_content_width + 2*frame_spc_hor,
		    m_content_height + 2*frame_spc_ver);
}

Mml::FrameType MmlMtableNode::frame() const
{
    QString value = explicitAttribute("frame", "none");
    return interpretFrameType(value, 0, 0);
}

Mml::FrameType MmlMtableNode::columnlines(int idx) const
{
    QString value = explicitAttribute("columnlines", "none");
    return interpretFrameType(value, idx, 0);
}

Mml::FrameType MmlMtableNode::rowlines(int idx) const
{
    QString value = explicitAttribute("rowlines", "none");
    return interpretFrameType(value, idx, 0);
}

void MmlMtableNode::paintSymbol(QPainter *p) const
{
    FrameType f = frame();
    if (f != FrameNone) {
	p->save();

	QPen pen = p->pen();
	if (f == FrameDashed)
	    pen.setStyle(Qt::DashLine);
	else
	    pen.setStyle(Qt::SolidLine);
	p->setPen(pen);
	p->drawRect(myRect());

	p->restore();
    }

    p->save();

    int col_spc = columnspacing();
    int row_spc = rowspacing();

    QPen pen = p->pen();
    int col_offset = 0;
    int i;
    for (i = 0; i < m_cell_size_data.numCols() - 1; ++i) {
	FrameType f = columnlines(i);
	col_offset += m_cell_size_data.col_widths[i];

	if (f != FrameNone) {
	    if (f == FrameDashed)
		pen.setStyle(Qt::DashLine);
	    else if (f == FrameSolid)
		pen.setStyle(Qt::SolidLine);

	    p->setPen(pen);
	    int x = col_offset + col_spc/2;
	    p->drawLine(x, -m_content_height/2, x, m_content_height/2 );
	}
	col_offset += col_spc;
    }

    int row_offset = 0;
    for (i = 0; i < m_cell_size_data.numRows() - 1; ++i) {
	FrameType f = rowlines(i);
	row_offset += m_cell_size_data.row_heights[i];

	if (f != FrameNone) {
	    if (f == FrameDashed)
		pen.setStyle(Qt::DashLine);
	    else if (f == FrameSolid)
		pen.setStyle(Qt::SolidLine);

	    p->setPen(pen);
	    int y = row_offset + row_spc/2 - m_content_height/2;
	    p->drawLine(0, y, m_content_width, y);
	}
	row_offset += row_spc;
    }

    p->restore();
}

int MmlMtableNode::framespacing_ver() const
{
    if (frame() == FrameNone)
	return (int)(0.2*em());

    QString value = explicitAttribute("framespacing", "0.4em 0.5ex");

    bool ok;
    FrameSpacing fs = interpretFrameSpacing(value, em(), ex(), &ok);
    if (ok)
	return fs.m_ver;
    else
	return (int)(0.5*ex());
}

int MmlMtableNode::framespacing_hor() const
{
    if (frame() == FrameNone)
	return (int)(0.2*em());

    QString value = explicitAttribute("framespacing", "0.4em 0.5ex");

    bool ok;
    FrameSpacing fs = interpretFrameSpacing(value, em(), ex(), &ok);
    if (ok)
	return fs.m_hor;
    else
	return (int)(0.4*em());
}

void MmlMtrNode::layoutCells(const QList<int> &col_widths,
				int col_spc)
{
    QRect mr = myRect();

    MmlNode *child = firstChild();
    int col_offset = 0;
    uint colnum = 0;
    for (; child != 0; child = child->nextSibling(), ++colnum) {
	Q_ASSERT(child->nodeType() == MtdNode);
	MmlMtdNode *mtd = (MmlMtdNode*) child;

	QRect r = QRect(0, mr.top(), col_widths[colnum], mr.height());
	mtd->setMyRect(r);
	mtd->setRelOrigin(QPoint(col_offset, 0));
	col_offset += col_widths[colnum] + col_spc;
    }

    updateMyRect();
}

int MmlMtdNode::scriptlevel(const MmlNode *child) const
{
    int sl = MmlNode::scriptlevel();
    if (child != 0 && child == firstChild())
	return sl + m_scriptlevel_adjust;
    else
	return sl;
}

void MmlMtdNode::setMyRect(const QRect &rect)
{
    MmlNode::setMyRect(rect);
    MmlNode *child = firstChild();
    if (child == 0)
	return;

    if (rect.width() < child->myRect().width()) {
	while (rect.width() < child->myRect().width()
		    && child->font().pointSize() > g_min_font_point_size) {

//    	    qWarning("MmlMtdNode::setMyRect(): rect.width()=%d, child()->myRect().width=%d sl=%d",
//	    		rect.width(), child->myRect().width(), m_scriptlevel_adjust);

	    ++m_scriptlevel_adjust;
	    child->layout();
	}

//    	qWarning("MmlMtdNode::setMyRect(): rect.width()=%d, child()->myRect().width=%d sl=%d",
//	    	    rect.width(), child->myRect().width(), m_scriptlevel_adjust);
    }

    QRect mr = myRect();
    QRect cmr = child->myRect();

    QPoint child_rel_origin;

    switch (columnalign()) {
	case ColAlignLeft:
	    child_rel_origin.setX(0);
	    break;
	case ColAlignCenter:
	    child_rel_origin.setX(mr.left() + (mr.width() - cmr.width())/2);
	    break;
	case ColAlignRight:
	    child_rel_origin.setX(mr.right() - cmr.width());
	    break;
    }

    switch (rowalign()) {
	case RowAlignTop:
	    child_rel_origin.setY(mr.top() - cmr.top());
	    break;
	case RowAlignCenter:
	case RowAlignBaseline:
	    child_rel_origin.setY(mr.top() -cmr.top() + (mr.height() - cmr.height())/2);
	    break;
	case RowAlignBottom:
	    child_rel_origin.setY(mr.bottom() - cmr.bottom());
	    break;
	case RowAlignAxis:
	    child_rel_origin.setY(0);
	    break;
    }

    child->setRelOrigin(child_rel_origin);
}

uint MmlMtdNode::colNum()
{
    MmlNode *syb = previousSibling();

    uint i = 0;
    for (; syb != 0; syb = syb->previousSibling())
	++i;

    return i;
}

uint MmlMtdNode::rowNum()
{
    MmlNode *row = parent()->previousSibling();

    uint i = 0;
    for (; row != 0; row = row->previousSibling())
	++i;

    return i;
}

MmlMtdNode::ColAlign MmlMtdNode::columnalign()
{
    QString val = explicitAttribute("columnalign");
    if (!val.isNull())
	return interpretColAlign(val, 0, 0);

    MmlNode *node = parent(); // <mtr>
    if (node == 0)
	return ColAlignCenter;

    uint colnum = colNum();
    val = node->explicitAttribute("columnalign");
    if (!val.isNull())
	return interpretColAlign(val, colnum, 0);

    node = node->parent(); // <mtable>
    if (node == 0)
	return ColAlignCenter;

    val = node->explicitAttribute("columnalign");
    if (!val.isNull())
	return interpretColAlign(val, colnum, 0);

    return ColAlignCenter;
}

MmlMtdNode::RowAlign MmlMtdNode::rowalign()
{
    QString val = explicitAttribute("rowalign");
    if (!val.isNull())
	return interpretRowAlign(val, 0, 0);

    MmlNode *node = parent(); // <mtr>
    if (node == 0)
	return RowAlignAxis;

    uint rownum = rowNum();
    val = node->explicitAttribute("rowalign");
    if (!val.isNull())
	return interpretRowAlign(val, rownum, 0);

    node = node->parent(); // <mtable>
    if (node == 0)
	return RowAlignAxis;

    val = node->explicitAttribute("rowalign");
    if (!val.isNull())
	return interpretRowAlign(val, rownum, 0);

    return RowAlignAxis;
}

void MmlMoverNode::layoutSymbol()
{
    MmlNode *base = firstChild();
    Q_ASSERT(base != 0);
    MmlNode *over = base->nextSibling();
    Q_ASSERT(over != 0);

    QRect base_rect = base->myRect();
    QRect over_rect = over->myRect();

    int spacing = (int)(g_mfrac_spacing*(over_rect.height()
                                         + base_rect.height()));

    base->setRelOrigin(QPoint(-base_rect.width()/2, 0));
    over->setRelOrigin(QPoint(-over_rect.width()/2,
                              base_rect.top() - spacing - over_rect.bottom()));
}

int MmlMoverNode::scriptlevel(const MmlNode *node) const
{
    MmlNode *base = firstChild();
    Q_ASSERT(base != 0);
    MmlNode *over = base->nextSibling();
    Q_ASSERT(over != 0);

    int sl = MmlNode::scriptlevel();
    if (node != 0 && node == over)
	return sl + 1;
    else
	return sl;
}

void MmlMunderNode::layoutSymbol()
{
    MmlNode *base = firstChild();
    Q_ASSERT(base != 0);
    MmlNode *under = base->nextSibling();
    Q_ASSERT(under != 0);

    QRect base_rect = base->myRect();
    QRect under_rect = under->myRect();

    int spacing = (int)(g_mfrac_spacing*(under_rect.height() + base_rect.height()));

    base->setRelOrigin(QPoint(-base_rect.width()/2, 0));
    under->setRelOrigin(QPoint(-under_rect.width()/2, base_rect.bottom() + spacing - under_rect.top()));
}

int MmlMunderNode::scriptlevel(const MmlNode *node) const
{
    MmlNode *base = firstChild();
    Q_ASSERT(base != 0);
    MmlNode *under = base->nextSibling();
    Q_ASSERT(under != 0);

    int sl = MmlNode::scriptlevel();
    if (node != 0 && node == under)
	return sl + 1;
    else
	return sl;
}

void MmlMunderoverNode::layoutSymbol()
{
    MmlNode *base = firstChild();
    Q_ASSERT(base != 0);
    MmlNode *under = base->nextSibling();
    Q_ASSERT(under != 0);
    MmlNode *over = under->nextSibling();
    Q_ASSERT(over != 0);

    QRect base_rect = base->myRect();
    QRect under_rect = under->myRect();
    QRect over_rect = over->myRect();

    int spacing = (int)(g_mfrac_spacing*(   base_rect.height()
						+ under_rect.height()
						+ over_rect.height())	);

    base->setRelOrigin(QPoint(-base_rect.width()/2, 0));
    under->setRelOrigin(QPoint(-under_rect.width()/2, base_rect.bottom() + spacing - under_rect.top()));
    over->setRelOrigin(QPoint(-over_rect.width()/2, base_rect.top() - spacing - under_rect.bottom()));
}

int MmlMunderoverNode::scriptlevel(const MmlNode *node) const
{
    MmlNode *base = firstChild();
    Q_ASSERT(base != 0);
    MmlNode *under = base->nextSibling();
    Q_ASSERT(under != 0);
    MmlNode *over = under->nextSibling();
    Q_ASSERT(over != 0);

    int sl = MmlNode::scriptlevel();
    if (node != 0 && (node == under || node == over))
	return sl + 1;
    else
	return sl;
}

int MmlMpaddedNode::interpretSpacing(QString value, int base_value, bool *ok) const
{
    if (ok != 0)
	*ok = false;

    value.replace(' ', "");

    QString sign, factor_str, pseudo_unit;
    bool percent = false;

    // extract the sign
    int idx = 0;
    if (idx < value.length() && (value.at(idx) == '+' || value.at(idx) == '-'))
	sign = value.at(idx++);

    // extract the factor
    while (idx < value.length() && (value.at(idx).isDigit() || value.at(idx) == '.'))
	factor_str.append(value.at(idx++));

    // extract the % sign
    if (idx < value.length() && value.at(idx) == '%') {
	percent = true;
	++idx;
    }

    // extract the pseudo-unit
    pseudo_unit = value.mid(idx);

    bool float_ok;
    double factor = factor_str.toFloat(&float_ok);
    if (!float_ok || factor < 0) {
	qWarning("MmlMpaddedNode::interpretSpacing(): could not parse \"%s\"", value.toLatin1().data());
	return 0;
    }

    if (percent)
	factor /= 100.0;

    QRect cr;
    if (firstChild() == 0)
	cr = QRect(0, 0, 0, 0);
    else
	cr = firstChild()->myRect();

    int unit_size;

    if (pseudo_unit.isEmpty())
	unit_size = base_value;
    else if (pseudo_unit == "width")
	unit_size = cr.width();
    else if (pseudo_unit == "height")
	unit_size = -cr.top();
    else if (pseudo_unit == "depth")
	unit_size = cr.bottom();
    else {
	bool unit_ok;
	unit_size = MmlNode::interpretSpacing("1" + pseudo_unit, &unit_ok);
	if (!unit_ok) {
	    qWarning("MmlMpaddedNode::interpretSpacing(): could not parse \"%s\"", value.toLatin1().data());
	    return 0;
	}
    }

    if (ok != 0)
	*ok = true;

    if (sign.isNull())
	return (int)(factor*unit_size);
    else if (sign == "+")
	return base_value + (int)(factor*unit_size);
    else // sign == "-"
	return base_value - (int)(factor*unit_size);
}

int MmlMpaddedNode::lspace() const
{
    QString value = explicitAttribute("lspace");

    if (value.isNull())
	return 0;

    bool ok;
    int lspace = interpretSpacing(value, 0, &ok);

    if (ok)
	return lspace;

    return 0;
}

int MmlMpaddedNode::width() const
{
    int child_width = 0;
    if (firstChild() != 0)
	child_width = firstChild()->myRect().width();

    QString value = explicitAttribute("width");
    if (value.isNull())
	return child_width;

    bool ok;
    int w = interpretSpacing(value, child_width, &ok);
    if (ok)
	return w;

    return child_width;
}

int MmlMpaddedNode::height() const
{
    QRect cr;
    if (firstChild() == 0)
	cr = QRect(0, 0, 0, 0);
    else
	cr = firstChild()->myRect();

    QString value = explicitAttribute("height");
    if (value.isNull())
	return -cr.top();

    bool ok;
    int h = interpretSpacing(value, -cr.top(), &ok);
    if (ok)
	return h;

    return -cr.top();
}

int MmlMpaddedNode::depth() const
{
    QRect cr;
    if (firstChild() == 0)
	cr = QRect(0, 0, 0, 0);
    else
	cr = firstChild()->myRect();

    QString value = explicitAttribute("depth");
    if (value.isNull())
	return cr.bottom();

    bool ok;
    int h = interpretSpacing(value, cr.bottom(), &ok);
    if (ok)
	return h;

    return cr.bottom();
}

void MmlMpaddedNode::layoutSymbol()
{
    MmlNode *child = firstChild();
    if (child == 0)
	return;

    child->setRelOrigin(QPoint(0, 0));
}


QRect MmlMpaddedNode::symbolRect() const
{
    return QRect(-lspace(), -height(), lspace() + width(), height() + depth());
}

// *******************************************************************
// QtMmlWidget
// *******************************************************************

/*!
    \class QtMmlWidget

    \brief The QtMmlWidget class renders mathematical formulas written in MathML 2.0.

    QtMmlWidget implements the Presentation Markup subset of the
    MathML 2.0 specification, with a few \link overview.html exceptions\endlink.

    \code
    QtMmlWidget *mmlWidget = new QtMmlWidget(this);
    QString errorMsg;
    int errorLine;
    int errorColumn;
    bool ok = mmlWidget->setContent(mmlText, &errorMsg, &errorLine, &errorColumn);
    if (!ok) {
	qWarning("MathML error: %s, Line: %d, Column: %d",
		 errorMsg.latin1(), errorLine, errorColumn);
    }
    \endcode

*/


/*!
    \enum QtMmlWidget::MmlFont

    This ennumerated type is used in fontName() and setFontName() to
    specify a font type.

    \value NormalFont	The default font type, used to render
    expressions for which no mathvariant or fontfamily is specified,
    or for which the "normal" mathvariant is specified.

    \value FrakturFont	The font type used to render expressions for
    which the "fraktur" mathvariant is specified.

    \value SansSerifFont    The font type used to render expressions
    for which the "sans-serif" mathvariant is specified.

    \value ScriptFont	    The font type used to render expressions
    for which the "script" mathvariant is specified.

    \value MonospaceFont    The font type used to render expressions
    for which the "monospace" mathvariant is specified.

    \value DoublestruckFont The font type used to render expressions
    for which the "doublestruck" mathvariant is specified.

    \sa setFontName() fontName() setBaseFontPointSize() baseFontPointSize()
*/

/*!
    Constructs a QtMmlWidget object. The \a parent
    parameter is passed to QFrame's constructor.
*/

QtMmlWidget::QtMmlWidget(QWidget *parent)
    : QFrame(parent)
{
    m_doc = new MmlDocument;
}

/*!
    Destructs a QtMmlWidget object.
*/

QtMmlWidget::~QtMmlWidget()
{
    delete m_doc;
}

/*!
    Returns the name of the font used to render the font \a type.

    \sa setFontName()  setBaseFontPointSize() baseFontPointSize() QtMmlWidget::MmlFont
*/

QString QtMmlWidget::fontName(MmlFont type) const
{
    return m_doc->fontName(type);
}

/*!
    Sets the name of the font used to render the font \a type to \a name.

    \sa fontName() setBaseFontPointSize() baseFontPointSize() QtMmlWidget::MmlFont
*/

void QtMmlWidget::setFontName(MmlFont type, const QString &name)
{
    m_doc->setFontName(type, name);
    m_doc->layout();
    update();
}

/*!
    If \a b is true, draws a red bounding rectangle around each
    expression; if \a b is false, no such rectangle is drawn.
    This is mostly useful for debugging MathML expressions.

    \sa drawFrames()
*/

void QtMmlWidget::setDrawFrames(bool b)
{
    g_draw_frames = b;
    update();
}

/*!
    Returns true if each expression should be drawn with a red
    bounding rectangle; otherwise returns false.
    This is mostly useful for debugging MathML expressions.

    \sa setDrawFrames()
*/

bool QtMmlWidget::drawFrames() const
{
    return g_draw_frames;
}

/*!
    Clears the contents of this widget.
*/
void QtMmlWidget::clear()
{
    m_doc->clear();
}

/*!
    Returns the point size of the font used to render expressions
    whose scriptlevel is 0.

    \sa setBaseFontPointSize() fontName() setFontName()
*/

int QtMmlWidget::baseFontPointSize() const
{
    return m_doc->baseFontPointSize();
}

/*!
    Sets the point \a size of the font used to render expressions
    whose scriptlevel is 0.

    \sa baseFontPointSize() fontName() setFontName()
*/

void QtMmlWidget::setBaseFontPointSize(int size)
{
    if (size < g_min_font_point_size)
	return;

    m_doc->setBaseFontPointSize(size);
    m_doc->layout();
    update();
}

/*!
    Returns the size of the formula in pixels.
*/

QSize QtMmlWidget::sizeHint() const
{
    QSize size = m_doc->size();
    if (size == QSize(0, 0))
    	return QSize(100, 50);
    return m_doc->size();
}

/*!
    Sets the MathML expression to be rendered. The expression is given
    in the string \a text. If the expression is successfully parsed,
    this method returns true; otherwise it returns false. If an error
    occured \a errorMsg is set to a diagnostic message, while \a
    errorLine and \a errorColumn contain the location of the error.
    Any of \a errorMsg, \a errorLine and \a errorColumn may be 0,
    in which case they are not set.

    \a text should contain MathML 2.0 presentation markup elements enclosed
    in a <math> element.
*/

bool QtMmlWidget::setContent(const QString &text, QString *errorMsg,
				int *errorLine, int *errorColumn)
{
    bool result = m_doc->setContent(text, errorMsg, errorLine, errorColumn);
    if (result)
	update();
    return result;
}

/*! \internal */

void QtMmlWidget::paintEvent(QPaintEvent *e)
{
    QFrame::paintEvent(e);
    QPainter p(this);
    if (e->rect().intersects(contentsRect()))
        p.setClipRegion(e->region().intersected(contentsRect()));

    QSize s = m_doc->size();
    int x = (width() - s.width())/2;
    int y = (height() - s.height())/2;
    m_doc->paint(&p, QPoint(x, y));
}

/*! \internal */

void QtMmlWidget::dump() const
{
    m_doc->dump();
}

// *******************************************************************
// Static helper functions
// *******************************************************************

static QString entityDeclarations()
{
    QString result = "<!DOCTYPE math [\n";

    const EntitySpec *ent = g_xml_entity_data;
    for (; ent->name != 0; ++ent) {
	result += "\t<!ENTITY " + QString(ent->name) + " \"" + ent->value + "\">\n";
    }

    result += "]>\n";

    return result;
}

static int interpretSpacing(QString value, int em, int ex, bool *ok)
{
    if (ok != 0)
	*ok = true;

    if (value == "thin")
	return 1;

    if (value == "medium")
	return 2;

    if (value == "thick")
	return 3;

    struct HSpacingValue {
	const char *name;
	float factor;
    };

    static const HSpacingValue g_h_spacing_data[] =
    {
	{ "veryverythinmathspace",	(float) 0.0555556   },
	{ "verythinmathspace",  	(float) 0.111111    },
	{ "thinmathspace",          	(float) 0.166667    },
	{ "mediummathspace",    	(float) 0.222222    },
	{ "thickmathspace",     	(float) 0.277778    },
	{ "verythickmathspace", 	(float) 0.333333    },
	{ "veryverythickmathspace", 	(float) 0.388889    },
	{ 0,    	    	    	(float) 0   	    }
    };

    const HSpacingValue *v = g_h_spacing_data;
    for (; v->name != 0; ++v) {
	if (value == v->name)
	    return (int)(em*v->factor);
    }

    if (value.endsWith("em")) {
	value.truncate(value.length() - 2);
	bool float_ok;
	float factor = value.toFloat(&float_ok);
	if (float_ok && factor >= 0)
	    return (int)(em*factor);
	else {
	    qWarning("interpretSpacing(): could not parse \"%sem\"", value.toLatin1().data());
	    if (ok != 0)
		*ok = false;
	    return 0;
	}
    }

    if (value.endsWith("ex")) {
	value.truncate(value.length() - 2);
	bool float_ok;
	float factor = value.toFloat(&float_ok);
	if (float_ok && factor >= 0)
	    return (int)(ex*factor);
	else {
	    qWarning("interpretSpacing(): could not parse \"%sex\"", value.toLatin1().data());
	    if (ok != 0)
		*ok = false;
	    return 0;
	}
    }

   if (value.endsWith("cm")) {
	value.truncate(value.length() - 2);
	bool float_ok;
	float factor = value.toFloat(&float_ok);
	if (float_ok && factor >= 0) {
            QScreen *screen = qApp->screenAt(QPoint());
            Q_ASSERT(screen != 0);
            return (int)(factor*10*screen->size().width()/screen->physicalSize().width());
	}
	else {
	    qWarning("interpretSpacing(): could not parse \"%scm\"", value.toLatin1().data());
	    if (ok != 0)
		*ok = false;
	    return 0;
	}
    }

    if (value.endsWith("mm")) {
	value.truncate(value.length() - 2);
	bool float_ok;
	float factor = value.toFloat(&float_ok);
	if (float_ok && factor >= 0) {
            QScreen *screen = qApp->screenAt(QPoint());
            Q_ASSERT(screen != 0);
            return (int)(factor*screen->size().width()/screen->physicalSize().width());
	}
	else {
	    qWarning("interpretSpacing(): could not parse \"%smm\"", value.toLatin1().data());
	    if (ok != 0)
		*ok = false;
	    return 0;
	}
    }

    if (value.endsWith("in")) {
	value.truncate(value.length() - 2);
	bool float_ok;
	float factor = value.toFloat(&float_ok);
	if (float_ok && factor >= 0) {
            QScreen *screen = qApp->screenAt(QPoint());
            Q_ASSERT(screen != 0);
            return (int)(factor*10*screen->size().width()/(2.54*screen->physicalSize().width()));
	}
	else {
	    qWarning("interpretSpacing(): could not parse \"%sin\"", value.toLatin1().data());
	    if (ok != 0)
		*ok = false;
	    return 0;
	}
    }

    if (value.endsWith("px")) {
	value.truncate(value.length() - 2);
	bool float_ok;
	int i = (int) value.toFloat(&float_ok);
	if (float_ok && i >= 0)
	    return i;
	else {
	    qWarning("interpretSpacing(): could not parse \"%spx\"", value.toLatin1().data());
	    if (ok != 0)
		*ok = false;
	    return 0;
	}
    }

    bool float_ok;
    int i = (int)value.toFloat(&float_ok);
    if (float_ok && i >= 0)
	return i;

    qWarning("interpretSpacing(): could not parse \"%s\"", value.toLatin1().data());
    if (ok != 0)
	*ok = false;
    return 0;
}

static int interpretPercentSpacing(QString value, int base, bool *ok)
{
    if (!value.endsWith("%")) {
	if (ok != 0)
	    *ok = false;
	return 0;
    }

    value.truncate(value.length() - 1);
    bool float_ok;
    float factor = value.toFloat(&float_ok);
    if (float_ok && factor >= 0) {
	if (ok != 0)
	    *ok = true;
	return (int)(base*factor/100.0);
    }

    qWarning("interpretPercentSpacing(): could not parse \"%s%%\"", value.toLatin1().data());
    if (ok != 0)
	*ok = false;
    return 0;
}

static int interpretPointSize(QString value, bool *ok)
{
    if (!value.endsWith("pt")) {
	if (ok != 0)
	    *ok = false;
	return 0;
    }

    value.truncate(value.length() - 2);
    bool float_ok;
    int pt_size = (int) value.toFloat(&float_ok);
    if (float_ok && pt_size > 0) {
	if (ok != 0)
	    *ok = true;
	return pt_size;
    }

    qWarning("interpretPointSize(): could not parse \"%spt\"", value.toLatin1().data());
    if (ok != 0)
	*ok = false;
    return 0;
}

static const NodeSpec *mmlFindNodeSpec(Mml::NodeType type)
{
    const NodeSpec *spec = g_node_spec_data;
    for (; spec->type != Mml::NoNode; ++spec) {
	if (type == spec->type) return spec;
    }
    return 0;
}

static const NodeSpec *mmlFindNodeSpec(const QString &tag)
{
    const NodeSpec *spec = g_node_spec_data;
    for (; spec->type != Mml::NoNode; ++spec) {
	if (tag == spec->tag) return spec;
    }
    return 0;
}

static bool mmlCheckChildType(Mml::NodeType parent_type, Mml::NodeType child_type,
			    QString *error_str)
{
    if (parent_type == Mml::UnknownNode || child_type == Mml::UnknownNode)
	return true;

    const NodeSpec *child_spec = mmlFindNodeSpec(child_type);
    const NodeSpec *parent_spec = mmlFindNodeSpec(parent_type);

    Q_ASSERT(parent_spec != 0);
    Q_ASSERT(child_spec != 0);

    QString allowed_child_types(parent_spec->child_types);
    // null list means any child type is valid
    if (allowed_child_types.isNull())
	return true;

    QString child_type_str = QString(" ") + child_spec->type_str + " ";
    if (!allowed_child_types.contains(child_type_str)) {
	if (error_str != 0)
	    *error_str = QString("illegal child ")
			  + child_spec->type_str
			  + " for parent "
			  + parent_spec->type_str;
	return false;
    }

    return true;
}

static bool mmlCheckAttributes(Mml::NodeType child_type, const MmlAttributeMap &attr,
			    QString *error_str)
{
    const NodeSpec *spec = mmlFindNodeSpec(child_type);
    Q_ASSERT(spec != 0);

    QString allowed_attr(spec->attributes);
    // empty list means any attr is valid
    if (allowed_attr.isEmpty())
	return true;

    MmlAttributeMap::const_iterator it = attr.begin(), end = attr.end();
    for (; it != end; ++it) {
	QString name = it.key();

	if (name.indexOf(':') != -1)
	    continue;

	QString padded_name = " " + name + " ";
	if (!allowed_attr.contains(padded_name)) {
	    if (error_str != 0)
		*error_str = QString("illegal attribute ")
				 + name
				 + " in "
				 + spec->type_str;
	    return false;
	}
    }

    return true;
}

static int attributeIndex(const QString &name)
{
    for (unsigned i = 0; i < g_oper_spec_rows; ++i) {
	if (name == g_oper_spec_names[i])
	    return i;
    }
    return -1;
}

static QString decodeEntityValue(QString literal)
{
    QString result;

    while (!literal.isEmpty()) {

	if (!literal.startsWith("&#")) {
	    qWarning("decodeEntityValue(): bad entity literal: \"%s\"", literal.toLatin1().data());
	    return QString();
	}

    literal = literal.mid(2);

	int i = literal.indexOf(';');
	if (i == -1) {
	    qWarning("decodeEntityValue(): bad entity literal: \"%s\"", literal.toLatin1().data());
	    return QString();
	}

	QString char_code = literal.left(i);
	literal = literal.mid(i + 1);

	if (char_code.isEmpty()) {
	    qWarning("decodeEntityValue(): bad entity literal: \"%s\"", literal.toLatin1().data());
	    return QString();
	}

    bool ok;
    unsigned int c;
    if (char_code.startsWith('x')) {
        char_code = char_code.mid(1);
        c = char_code.toUInt(&ok, 16);
    } else {
        c = char_code.toUInt(&ok, 10);
    }

    if (!ok || c > 0x10FFFF) {
        qWarning("decodeEntityValue(): bad entity literal: \"%s\"", literal.toLatin1().data());
        return QString();
    }

    if (c <= 0xFFFF) {
        result += QChar(c);
    } else {
        c -= 0x10000;
        result += QChar(0xD800 + (c >> 10));
        result += QChar(0xDC00 + (c & 0x3FF));
    }
    }

    return result;
}

static const EntitySpec *searchEntitySpecData(const QString &value, const EntitySpec *from = 0)
{
    const EntitySpec *ent = from;
    if (ent == 0)
	ent = g_xml_entity_data;
    for (; ent->name != 0; ++ent) {
	QString ent_value = decodeEntityValue(ent->value);
	if (value == ent_value)
	    return ent;
    }
    return 0;
}

struct OperSpecSearchResult
{
    OperSpecSearchResult() { prefix_form = infix_form = postfix_form = 0; }

    const OperSpec *prefix_form,
		   *infix_form,
		   *postfix_form;

    const OperSpec *&getForm(Mml::FormType f);
    bool haveForm(Mml::FormType f)
	{ return getForm(f) != 0; }
    void addForm(const OperSpec *spec)
	{ getForm(spec->form) = spec; }
};

const OperSpec *&OperSpecSearchResult::getForm(Mml::FormType f)
{
    switch (f) {
	case Mml::PrefixForm:
	    return prefix_form;
	case Mml::InfixForm:
	    return infix_form;
	case Mml::PostfixForm:
	    return postfix_form;
    }
    return postfix_form; // just to avoid warning
}

/*
    Searches g_oper_spec_data and returns any instance of operator name. There may
    be more instances, but since the list is sorted, they will be next to each other.
*/
static const OperSpec *searchOperSpecData(const QString &name)
{
    const char *name_latin1 = name.toLatin1().data();

    // binary search
    // establish invariant g_oper_spec_data[begin].name < name < g_oper_spec_data[end].name

    int cmp = qstrcmp(name_latin1, g_oper_spec_data[0].name);
    if (cmp < 0)
    	return 0;

    if (cmp == 0)
    	return g_oper_spec_data;

    uint begin = 0;
    uint end = g_oper_spec_count;

    // invariant holds
    while (end - begin > 1) {
    	uint mid = (begin + end)/2;

    	const OperSpec *spec = g_oper_spec_data + mid;
	int cmp = qstrcmp(name_latin1, spec->name);
    	if (cmp < 0)
	    end = mid;
	else if (cmp > 0)
	    begin = mid;
	else
	    return spec;
    }

    return 0;
}

/*
    This searches g_oper_spec_data until at least one name in name_list is found with FormType form,
    or until name_list is exhausted. The idea here is that if we don't find the operator in the
    specified form, we still want to use some other available form of that operator.
*/
static OperSpecSearchResult _mmlFindOperSpec(const QStringList &name_list, Mml::FormType form)
{
    OperSpecSearchResult result;

    QStringList::const_iterator it = name_list.begin();
    for (; it != name_list.end(); ++it) {
    	const QString &name = *it;

    	const OperSpec *spec = searchOperSpecData(name);

	if (spec == 0)
	    continue;

    	const char *name_latin1 = name.toLatin1().data();

    	// backtrack to the first instance of name
	while (spec > g_oper_spec_data && qstrcmp((spec - 1)->name, name_latin1) == 0)
    	    --spec;

    	// iterate over instances of name until the instances are exhausted or until we
	// find an instance in the specified form.
	do {
	    result.addForm(spec++);
	    if (result.haveForm(form))
	    	break;
	} while (qstrcmp(spec->name, name_latin1) == 0);

	if (result.haveForm(form))
	    break;
    }

    return result;
}

/*
    text is a string between <mo> and </mo>. It can be a character ('+'), an
    entity reference ("&infin;") or a character reference ("&#x0221E"). Our
    job is to find an operator spec in the operator dictionary (g_oper_spec_data)
    that matches text. Things are further complicated by the fact, that many
    operators come in several forms (prefix, infix, postfix).

    If available, this function returns an operator spec matching text in the specified
    form. If such operator is not available, returns an operator spec that matches
    text, but of some other form in the preference order specified by the MathML spec.
    If that's not available either, returns the default operator spec.
*/
static const OperSpec *mmlFindOperSpec(const QString &text, Mml::FormType form)
{
    QStringList name_list;
    name_list.append(text);

    // First, just try to find text in the operator dictionary.
    OperSpecSearchResult result = _mmlFindOperSpec(name_list, form);

    if (!result.haveForm(form)) {
    	// Try to find other names for the operator represented by text.

	const EntitySpec *ent = 0;
	for (;;) {
	    ent = searchEntitySpecData(text, ent);
	    if (ent == 0)
		break;
	    name_list.append('&' + QString(ent->name) + ';');
	    ++ent;
	}

	result = _mmlFindOperSpec(name_list, form);
    }

    const OperSpec *spec = result.getForm(form);
    if (spec != 0)
	return spec;

    spec = result.getForm(Mml::InfixForm);
    if (spec != 0)
	return spec;

    spec = result.getForm(Mml::PostfixForm);
    if (spec != 0)
	return spec;

    spec = result.getForm(Mml::PrefixForm);
    if (spec != 0)
	return spec;

    return &g_oper_spec_defaults;
}

static QString mmlDictAttribute(const QString &name, const OperSpec *spec)
{
    int i = attributeIndex(name);
    if (i == -1)
	return QString();
    else
	return spec->attributes[i];
}

static uint interpretMathVariant(const QString &value, bool *ok)
{
    struct MathVariantValue {
	const char *value;
	uint mv;
    };

    static const MathVariantValue g_mv_data[] =
    {
	{ "normal",     	    	    Mml::NormalMV },
	{ "bold",   	    	    	    Mml::BoldMV },
	{ "italic",   	    	    	    Mml::ItalicMV },
	{ "bold-italic",   	    	    Mml::BoldMV | Mml::ItalicMV },
	{ "double-struck",   	    	    Mml::DoubleStruckMV },
	{ "bold-fraktur",   	    	    Mml::BoldMV | Mml::FrakturMV },
	{ "script",   	    	    	    Mml::ScriptMV },
	{ "bold-script",	    	    Mml::BoldMV | Mml::ScriptMV },
	{ "fraktur",   	    	    	    Mml::FrakturMV },
	{ "sans-serif",   	    	    Mml::SansSerifMV },
	{ "bold-sans-serif",   	    	    Mml::BoldMV | Mml::SansSerifMV },
	{ "sans-serif-italic",   	    Mml::SansSerifMV | Mml::ItalicMV },
	{ "sans-serif-bold-italic",         Mml::SansSerifMV | Mml::ItalicMV | Mml::BoldMV },
	{ "monospace",   	    	    Mml::MonospaceMV },
	{ 0,    	    	    	    0 }
    };

    const MathVariantValue *v = g_mv_data;
    for (; v->value != 0; ++v) {
	if (value == v->value) {
	    if (ok != 0)
		*ok = true;
	    return v->mv;
	}
    }

    if (ok != 0)
	*ok = false;

   qWarning("interpretMathVariant(): could not parse value: \"%s\"", value.toLatin1().data());

    return Mml::NormalMV;
}

static Mml::FormType interpretForm(const QString &value, bool *ok)
{
    if (ok != 0)
	*ok = true;

    if (value == "prefix")
	return Mml::PrefixForm;
    if (value == "infix")
	return Mml::InfixForm;
    if (value == "postfix")
	return Mml::PostfixForm;

    if (ok != 0)
	*ok = false;

    qWarning("interpretForm(): could not parse value \"%s\"", value.toLatin1().data());
    return Mml::InfixForm;
}

static Mml::ColAlign interpretColAlign(const QString &value_list, uint colnum, bool *ok)
{
    QString value = interpretListAttr(value_list, colnum, "center");

    if (ok != 0)
	*ok = true;

    if (value == "left")
	return Mml::ColAlignLeft;
    if (value == "right")
	return Mml::ColAlignRight;
    if (value == "center")
	return Mml::ColAlignCenter;

    if (ok != 0)
	*ok = false;

    qWarning("interpretColAlign(): could not parse value \"%s\"", value.toLatin1().data());
    return Mml::ColAlignCenter;
}

static Mml::RowAlign interpretRowAlign(const QString &value_list, uint rownum, bool *ok)
{
    QString value = interpretListAttr(value_list, rownum, "axis");

    if (ok != 0)
	*ok = true;

    if (value == "top")
	return Mml::RowAlignTop;
    if (value == "center")
	return Mml::RowAlignCenter;
    if (value == "bottom")
	return Mml::RowAlignBottom;
    if (value == "baseline")
	return Mml::RowAlignBaseline;
    if (value == "axis")
	return Mml::RowAlignAxis;

    if (ok != 0)
	*ok = false;

    qWarning("interpretRowAlign(): could not parse value \"%s\"", value.toLatin1().data());
    return Mml::RowAlignAxis;
}

static QString interpretListAttr(const QString &value_list, int idx, const QString &def)
{
    QStringList l = value_list.split(' ');

    if (l.count() == 0)
	return def;

    if (l.count() <= idx)
	return l[l.count() - 1];
    else
	return l[idx];
}

static Mml::FrameType interpretFrameType(const QString &value_list, uint idx, bool *ok)
{
    if (ok != 0)
	*ok = true;

    QString value = interpretListAttr(value_list, idx, "none");

    if (value == "none")
	return Mml::FrameNone;
    if (value == "solid")
	return Mml::FrameSolid;
    if (value == "dashed")
	return Mml::FrameDashed;

    if (ok != 0)
	*ok = false;

    qWarning("interpretFrameType(): could not parse value \"%s\"", value.toLatin1().data());
    return Mml::FrameNone;
}


static Mml::FrameSpacing interpretFrameSpacing(const QString &value_list, int em, int ex, bool *ok)
{
    Mml::FrameSpacing fs;

    QStringList l = value_list.split(' ');
    if (l.count() != 2) {
	qWarning("interpretFrameSpacing: could not parse value \"%s\"", value_list.toLatin1().data());
	if (ok != 0)
	    *ok = false;
	return Mml::FrameSpacing((int)(0.4*em), (int)(0.5*ex));
    }

    bool hor_ok, ver_ok;
    fs.m_hor = interpretSpacing(l[0], em, ex, &hor_ok);
    fs.m_ver = interpretSpacing(l[1], em, ex, &ver_ok);

    if (ok != 0)
	*ok = hor_ok && ver_ok;

    return fs;
}

static QFont interpretDepreciatedFontAttr(const MmlAttributeMap &font_attr, QFont &fn, int em, int ex)
{
    if (font_attr.contains("fontsize")) {
	QString value = font_attr["fontsize"];

	for (;;) {

	    bool ok;
	    int ptsize = interpretPointSize(value, &ok);
	    if (ok) {
		fn.setPointSize(ptsize);
		break;
	    }

	    ptsize = interpretPercentSpacing(value, fn.pointSize(), &ok);
	    if (ok) {
		fn.setPointSize(ptsize);
		break;
	    }

	    int size = interpretSpacing(value, em, ex, &ok);
	    if (ok) {
		fn.setPixelSize(size);
		break;
	    }

	    break;
	}
    }

    if (font_attr.contains("fontweight")) {
	QString value = font_attr["fontweight"];
	if (value == "normal")
	    fn.setBold(false);
	else if (value == "bold")
	    fn.setBold(true);
	else
	    qWarning("interpretDepreciatedFontAttr(): could not parse fontweight \"%s\"", value.toLatin1().data());
    }

    if (font_attr.contains("fontstyle")) {
	QString value = font_attr["fontstyle"];
	if (value == "normal")
	    fn.setItalic(false);
	else if (value == "italic")
	    fn.setItalic(true);
	else
	    qWarning("interpretDepreciatedFontAttr(): could not parse fontstyle \"%s\"", value.toLatin1().data());
    }

    if (font_attr.contains("fontfamily")) {
	QString value = font_attr["fontfamily"];
	fn.setFamily(value);
    }

    return fn;
}

static QFont interpretMathSize(QString value, QFont &fn, int em, int ex, bool *ok)
{
    if (ok != 0)
	*ok = true;

    if (value == "small") {
	fn.setPointSize((int)(fn.pointSize()*0.7));
	return fn;
    }

    if (value == "normal")
	return fn;

    if (value == "big") {
	fn.setPointSize((int)(fn.pointSize()*1.5));
	return fn;
    }

    bool size_ok;

    int ptsize = interpretPointSize(value, &size_ok);
    if (size_ok) {
	fn.setPointSize(ptsize);
	return fn;
    }

    int size = interpretSpacing(value, em, ex, &size_ok);
    if (size_ok) {
	fn.setPixelSize(size);
	return fn;
    }

    if (ok != 0)
	*ok = false;
    qWarning("interpretMathSize(): could not parse mathsize \"%s\"", value.toLatin1().data());
    return fn;
}

/*!
    \class QtMmlDocument

    \brief The QtMmlDocument class renders mathematical formulas written in MathML 2.0.

    This class provides a direct API to the rendering engine used by
    QtMmlWidget. It can be used to paint MathML inside other widgets.

    All methods work the same as the corresponding methods in QtMmlWidget.
*/

/*!
  Constructs an empty MML document.
*/
QtMmlDocument::QtMmlDocument()
{
    m_doc = new MmlDocument;
}

/*!
  Destroys the MML document.
*/
QtMmlDocument::~QtMmlDocument()
{
    delete m_doc;
}

/*!
    Clears the contents of this MML document.
*/
void QtMmlDocument::clear()
{
    m_doc->clear();
}

/*!
    Sets the MathML expression to be rendered. The expression is given
    in the string \a text. If the expression is successfully parsed,
    this method returns true; otherwise it returns false. If an error
    occured \a errorMsg is set to a diagnostic message, while \a
    errorLine and \a errorColumn contain the location of the error.
    Any of \a errorMsg, \a errorLine and \a errorColumn may be 0,
    in which case they are not set.

    \a text should contain MathML 2.0 presentation markup elements enclosed
    in a <math> element.
*/
bool QtMmlDocument::setContent(QString text, QString *errorMsg,
                               int *errorLine, int *errorColumn)
{
    return m_doc->setContent(text, errorMsg, errorLine, errorColumn);
}

/*!
  Renders this MML document with the painter \a p at position \a pos.
*/
void QtMmlDocument::paint(QPainter *p, const QPoint &pos) const
{
    m_doc->paint(p, pos);
}

/*!
    Returns the size of this MML document, as rendered, in pixels.
*/
QSize QtMmlDocument::size() const
{
    return m_doc->size();
}

/*!
    Returns the name of the font used to render the font \a type.

    \sa setFontName()  setBaseFontPointSize() baseFontPointSize() QtMmlWidget::MmlFont
*/
QString QtMmlDocument::fontName(QtMmlWidget::MmlFont type) const
{
    return m_doc->fontName(type);
}

/*!
    Sets the name of the font used to render the font \a type to \a name.

    \sa fontName() setBaseFontPointSize() baseFontPointSize() QtMmlWidget::MmlFont
*/
void QtMmlDocument::setFontName(QtMmlWidget::MmlFont type, const QString &name)
{
    m_doc->setFontName(type, name);
    m_doc->layout();
}

/*!
    Returns the point size of the font used to render expressions
    whose scriptlevel is 0.

    \sa setBaseFontPointSize() fontName() setFontName()
*/
int QtMmlDocument::baseFontPointSize() const
{
    return m_doc->baseFontPointSize();
}

/*!
    Sets the point \a size of the font used to render expressions
    whose scriptlevel is 0.

    \sa baseFontPointSize() fontName() setFontName()
*/
void QtMmlDocument::setBaseFontPointSize(int size)
{
    m_doc->setBaseFontPointSize(size);
    m_doc->layout();
}
