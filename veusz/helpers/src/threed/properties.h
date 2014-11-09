#ifndef PROPERTIES_H
#define PROPERTIES_H

struct SurfaceProp
{
  SurfaceProp(float _r=0.5f, float _g=0.5f, float _b=0.5f,
	      float _specular=0.5f, float _diffuse=0.5f, float _trans=0,
	      bool _hide=0)
    : r(_r), g(_g), b(_b),
      specular(_specular), diffuse(_diffuse), trans(_trans),
      hide(_hide), _ref_cnt(0)
  {
  }

  float r, g, b;
  float specular, diffuse, trans;
  bool hide;

  // used to reference count usages by Object() instances
  mutable unsigned _ref_cnt;
};

struct LineProp
{
  LineProp(float _r=0, float _g=0, float _b=0,
	   float _specular=0.5f, float _diffuse=0.5f, float _trans=0,
	   float _width=1, bool _hide=0)
    : r(_r), g(_g), b(_b),
      specular(_specular), diffuse(_diffuse), trans(_trans),
      width(_width), hide(_hide), _ref_cnt(0)
  {
  }

  float r, g, b;
  float specular, diffuse, trans;
  float width;
  bool hide;

  // used to reference count usages by Object() instances
  mutable unsigned _ref_cnt;
};

//#include <stdio.h>
template<class T>
class PropSmartPtr
{
public:
  PropSmartPtr(T* p)
    : p_(p)
  {
    if(p_ != 0)
      {
	++p_->_ref_cnt;
	//printf("prop: %p +1 -> %i\n", p_, p_->_ref_cnt);
      }
  }

  ~PropSmartPtr()
  {
    if(p_ != 0)
      {
	--p_->_ref_cnt;
	//printf("prop: %p -1 -> %i\n", p_, p_->_ref_cnt);
	if(p_->_ref_cnt == 0)
	  delete p_;
      }
  }

  PropSmartPtr(const PropSmartPtr<T> &r)
    : p_(r.p_)
  {
    if(p_ != 0)
      {
	++p_->_ref_cnt;
	//printf("prop: %p +1 -> %i\n", p_, p_->_ref_cnt);
      }
  }

  T* operator->() { return p_; }
  const T* ptr() const { return p_; }

private:
  T* p_;
};

#endif
