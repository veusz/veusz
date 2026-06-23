#include "latex.h"
#include "platform/qt/graphic_qt.h"

#include <QByteArray>
#include <QColor>
#include <QCoreApplication>
#include <QPainter>
#include <QRect>
#include <QSize>
#include <QTemporaryFile>
#include <QString>
#include <QtSvg/QSvgGenerator>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace {

bool g_initialized = false;
std::mutex g_mutex;

tex::color parseColor(const char* s, tex::color fallback) {
  if (!s || !*s) return fallback;
  QColor q(QString::fromUtf8(s));
  if (!q.isValid()) return fallback;
  return q.rgba();
}

char* dupString(const std::string& s) {
  char* p = static_cast<char*>(std::malloc(s.size() + 1));
  if (!p) return nullptr;
  std::memcpy(p, s.c_str(), s.size() + 1);
  return p;
}

}  // namespace

extern "C" int microtex_render_svg(
    const char* tex_utf8,
    float text_size,
    int width,
    const char* foreground,
    const char* background,
    char** out_svg,
    size_t* out_len,
    char** out_error) {
  std::lock_guard<std::mutex> guard(g_mutex);
  if (out_svg) *out_svg = nullptr;
  if (out_len) *out_len = 0;
  if (out_error) *out_error = nullptr;

  if (!tex_utf8) {
    if (out_error) *out_error = dupString("tex_utf8 is null");
    return 1;
  }
  if (!QCoreApplication::instance()) {
    if (out_error) *out_error = dupString("QCoreApplication instance is required");
    return 2;
  }

  try {
    if (!g_initialized) {
      const char* res_root = std::getenv("VEUSZ_MICROTEX_RES");
      if (res_root && *res_root) {
        tex::LaTeX::init(res_root);
      } else {
        tex::LaTeX::init();
      }
      g_initialized = true;
    }

    auto render = tex::LaTeX::parse(
        QString::fromUtf8(tex_utf8).toStdWString(),
        width,
        text_size,
        text_size / 3.f,
        parseColor(foreground, tex::black));

    if (!render) {
      if (out_error) *out_error = dupString("MicroTeX render returned null");
      return 3;
    }

    const int w = std::max(1, render->getWidth());
    const int h = std::max(1, render->getHeight());

    QTemporaryFile tmp;
    tmp.setAutoRemove(true);
    if (!tmp.open()) {
      if (out_error) *out_error = dupString("cannot create temporary file");
      return 4;
    }

    QSvgGenerator gen;
    gen.setFileName(tmp.fileName());
    gen.setSize(QSize(w, h));
    gen.setViewBox(QRect(0, 0, w, h));
    gen.setTitle(QStringLiteral("MicroTeX"));
    gen.setDescription(QStringLiteral("MicroTeX SVG output"));

    QPainter painter(&gen);
    painter.fillRect(0, 0, w, h, QColor::fromRgba(parseColor(background, 0)));
    tex::Graphics2D_qt g2(&painter);
    render->draw(g2, 0, 0);
    painter.end();

    if (!tmp.seek(0)) {
      if (out_error) *out_error = dupString("cannot rewind SVG temp file");
      return 5;
    }

    QByteArray svg = tmp.readAll();
    if (svg.isEmpty()) {
      if (out_error) *out_error = dupString("empty SVG output");
      return 6;
    }

    if (out_svg) {
      *out_svg = static_cast<char*>(std::malloc(static_cast<size_t>(svg.size()) + 1));
      if (!*out_svg) {
        if (out_error) *out_error = dupString("allocation failed");
        return 7;
      }
      std::memcpy(*out_svg, svg.constData(), static_cast<size_t>(svg.size()));
      (*out_svg)[svg.size()] = '\0';
    }
    if (out_len) *out_len = static_cast<size_t>(svg.size());
    return 0;
  } catch (const std::exception& e) {
    if (out_error) *out_error = dupString(e.what());
    return 100;
  } catch (...) {
    if (out_error) *out_error = dupString("unknown error");
    return 101;
  }
}

extern "C" void microtex_free(void* p) {
  std::free(p);
}
