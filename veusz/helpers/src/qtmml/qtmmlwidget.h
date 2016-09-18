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

#ifndef QTMMLWIDGET_H
#define QTMMLWIDGET_H

#include <QtGui/QFrame>
#include <QtXml/QtXml>

class MmlDocument;

#if defined(Q_WS_WIN)
#  if !defined(QT_QTMMLWIDGET_EXPORT) && !defined(QT_QTMMLWIDGET_IMPORT)
#    define QT_QTMMLWIDGET_EXPORT
#  elif defined(QT_QTMMLWIDGET_IMPORT)
#    if defined(QT_QTMMLWIDGET_EXPORT)
#      undef QT_QTMMLWIDGET_EXPORT
#    endif
#    define QT_QTMMLWIDGET_EXPORT __declspec(dllimport)
#  elif defined(QT_QTMMLWIDGET_EXPORT)
#    undef QT_QTMMLWIDGET_EXPORT
#    define QT_QTMMLWIDGET_EXPORT __declspec(dllexport)
#  endif
#else
#  define QT_QTMMLWIDGET_EXPORT
#endif

class QT_QTMMLWIDGET_EXPORT QtMmlWidget : public QFrame
{
    public:
	enum MmlFont { NormalFont, FrakturFont, SansSerifFont, ScriptFont,
				    MonospaceFont, DoublestruckFont };

	QtMmlWidget(QWidget *parent = 0);
	~QtMmlWidget();

	QString fontName(MmlFont type) const;
	void setFontName(MmlFont type, const QString &name);
	int baseFontPointSize() const;
	void setBaseFontPointSize(int size);

	bool setContent(const QString &text, QString *errorMsg = 0,
			    int *errorLine = 0, int *errorColumn = 0);
	void dump() const;
	virtual QSize sizeHint() const;

	void setDrawFrames(bool b);
	bool drawFrames() const;

	void clear();

    protected:
	virtual void paintEvent(QPaintEvent *e);

    private:
	MmlDocument *m_doc;
};


class QT_QTMMLWIDGET_EXPORT QtMmlDocument
{
public:
    QtMmlDocument();
    ~QtMmlDocument();
    void clear();

    bool setContent(QString text, QString *errorMsg = 0,
                    int *errorLine = 0, int *errorColumn = 0);
    void paint(QPainter *p, const QPoint &pos) const;
    QSize size() const;

    QString fontName(QtMmlWidget::MmlFont type) const;
    void setFontName(QtMmlWidget::MmlFont type, const QString &name);

    int baseFontPointSize() const;
    void setBaseFontPointSize(int size);
private:
    MmlDocument *m_doc;
};

#endif
