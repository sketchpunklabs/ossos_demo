class Vec3 extends Array {
  // #region STATIC PROPERTIES
  static UP = [0, 1, 0];
  static DOWN = [0, -1, 0];
  static LEFT = [-1, 0, 0];
  static RIGHT = [1, 0, 0];
  static FORWARD = [0, 0, 1];
  static BACK = [0, 0, -1];
  constructor(v, y, z) {
    super(3);
    if (v instanceof Vec3 || v instanceof Float32Array || v instanceof Array && v.length == 3) {
      this[0] = v[0];
      this[1] = v[1];
      this[2] = v[2];
    } else if (typeof v === "number" && typeof y === "number" && typeof z === "number") {
      this[0] = v;
      this[1] = y;
      this[2] = z;
    } else if (typeof v === "number") {
      this[0] = v;
      this[1] = v;
      this[2] = v;
    } else {
      this[0] = 0;
      this[1] = 0;
      this[2] = 0;
    }
  }
  // #endregion
  // #region GETTERS
  get len() {
    return Math.sqrt(this[0] ** 2 + this[1] ** 2 + this[2] ** 2);
  }
  get lenSqr() {
    return this[0] ** 2 + this[1] ** 2 + this[2] ** 2;
  }
  get isZero() {
    return this[0] === 0 && this[1] === 0 && this[2] === 0;
  }
  clone() {
    return new Vec3(this);
  }
  // #endregion
  // #region SETTERS
  xyz(x, y, z) {
    this[0] = x;
    this[1] = y;
    this[2] = z;
    return this;
  }
  copy(a) {
    this[0] = a[0];
    this[1] = a[1];
    this[2] = a[2];
    return this;
  }
  copyTo(a) {
    a[0] = this[0];
    a[1] = this[1];
    a[2] = this[2];
    return this;
  }
  setInfinite(sign = 1) {
    this[0] = Infinity * sign;
    this[1] = Infinity * sign;
    this[2] = Infinity * sign;
    return this;
  }
  /** Generate a random vector. Can choose per axis range */
  rnd(x0 = 0, x1 = 1, y0 = 0, y1 = 1, z0 = 0, z1 = 1) {
    let t;
    t = Math.random();
    this[0] = x0 * (1 - t) + x1 * t;
    t = Math.random();
    this[1] = y0 * (1 - t) + y1 * t;
    t = Math.random();
    this[2] = z0 * (1 - t) + z1 * t;
    return this;
  }
  // #endregion
  // #region FROM OPERATORS
  fromAdd(a, b) {
    this[0] = a[0] + b[0];
    this[1] = a[1] + b[1];
    this[2] = a[2] + b[2];
    return this;
  }
  fromSub(a, b) {
    this[0] = a[0] - b[0];
    this[1] = a[1] - b[1];
    this[2] = a[2] - b[2];
    return this;
  }
  fromMul(a, b) {
    this[0] = a[0] * b[0];
    this[1] = a[1] * b[1];
    this[2] = a[2] * b[2];
    return this;
  }
  fromScale(a, s) {
    this[0] = a[0] * s;
    this[1] = a[1] * s;
    this[2] = a[2] * s;
    return this;
  }
  fromScaleThenAdd(scale, a, b) {
    this[0] = a[0] * scale + b[0];
    this[1] = a[1] * scale + b[1];
    this[2] = a[2] * scale + b[2];
    return this;
  }
  fromCross(a, b) {
    const ax = a[0], ay = a[1], az = a[2], bx = b[0], by = b[1], bz = b[2];
    this[0] = ay * bz - az * by;
    this[1] = az * bx - ax * bz;
    this[2] = ax * by - ay * bx;
    return this;
  }
  fromNorm(a) {
    let mag = Math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2);
    if (mag != 0) {
      mag = 1 / mag;
      this[0] = a[0] * mag;
      this[1] = a[1] * mag;
      this[2] = a[2] * mag;
    } else {
      this[0] = 0;
      this[1] = 0;
      this[2] = 0;
    }
    return this;
  }
  fromNegate(a) {
    this[0] = -a[0];
    this[1] = -a[1];
    this[2] = -a[2];
    return this;
  }
  fromInvert(a) {
    this[0] = 1 / a[0];
    this[1] = 1 / a[1];
    this[2] = 1 / a[2];
    return this;
  }
  fromQuat(q, v = [0, 0, 1]) {
    return this.copy(v).transformQuat(q);
  }
  fromLerp(a, b, t) {
    const ti = 1 - t;
    this[0] = a[0] * ti + b[0] * t;
    this[1] = a[1] * ti + b[1] * t;
    this[2] = a[2] * ti + b[2] * t;
    return this;
  }
  fromSlerp(a, b, t) {
    const angle = Math.acos(Math.min(Math.max(Vec3.dot(a, b), -1), 1));
    const sin = Math.sin(angle);
    const ta = Math.sin((1 - t) * angle) / sin;
    const tb = Math.sin(t * angle) / sin;
    this[0] = ta * a[0] + tb * b[0];
    this[1] = ta * a[1] + tb * b[1];
    this[2] = ta * a[2] + tb * b[2];
    return this;
  }
  // #endregion
  // #region LOADING / CONVERSION
  /** Used to get data from a flat buffer */
  fromBuf(ary, idx) {
    this[0] = ary[idx];
    this[1] = ary[idx + 1];
    this[2] = ary[idx + 2];
    return this;
  }
  /** Put data into a flat buffer */
  toBuf(ary, idx) {
    ary[idx] = this[0];
    ary[idx + 1] = this[1];
    ary[idx + 2] = this[2];
    return this;
  }
  // #endregion
  // #region OPERATORS
  add(a) {
    this[0] += a[0];
    this[1] += a[1];
    this[2] += a[2];
    return this;
  }
  sub(v) {
    this[0] -= v[0];
    this[1] -= v[1];
    this[2] -= v[2];
    return this;
  }
  mul(v) {
    this[0] *= v[0];
    this[1] *= v[1];
    this[2] *= v[2];
    return this;
  }
  scale(v) {
    this[0] *= v;
    this[1] *= v;
    this[2] *= v;
    return this;
  }
  divScale(v) {
    this[0] /= v;
    this[1] /= v;
    this[2] /= v;
    return this;
  }
  addScaled(a, s) {
    this[0] += a[0] * s;
    this[1] += a[1] * s;
    this[2] += a[2] * s;
    return this;
  }
  invert() {
    this[0] = 1 / this[0];
    this[1] = 1 / this[1];
    this[2] = 1 / this[2];
    return this;
  }
  norm() {
    let mag = Math.sqrt(this[0] ** 2 + this[1] ** 2 + this[2] ** 2);
    if (mag != 0) {
      mag = 1 / mag;
      this[0] *= mag;
      this[1] *= mag;
      this[2] *= mag;
    }
    return this;
  }
  cross(b) {
    const ax = this[0], ay = this[1], az = this[2], bx = b[0], by = b[1], bz = b[2];
    this[0] = ay * bz - az * by;
    this[1] = az * bx - ax * bz;
    this[2] = ax * by - ay * bx;
    return this;
  }
  abs() {
    this[0] = Math.abs(this[0]);
    this[1] = Math.abs(this[1]);
    this[2] = Math.abs(this[2]);
    return this;
  }
  floor() {
    this[0] = Math.floor(this[0]);
    this[1] = Math.floor(this[1]);
    this[2] = Math.floor(this[2]);
    return this;
  }
  ceil() {
    this[0] = Math.ceil(this[0]);
    this[1] = Math.ceil(this[1]);
    this[2] = Math.ceil(this[2]);
    return this;
  }
  min(a) {
    this[0] = Math.min(this[0], a[0]);
    this[1] = Math.min(this[1], a[1]);
    this[2] = Math.min(this[2], a[2]);
    return this;
  }
  max(a) {
    this[0] = Math.max(this[0], a[0]);
    this[1] = Math.max(this[1], a[1]);
    this[2] = Math.max(this[2], a[2]);
    return this;
  }
  /** When values are very small, like less then 0.000001, just make it zero */
  nearZero() {
    if (Math.abs(this[0]) <= 1e-6)
      this[0] = 0;
    if (Math.abs(this[1]) <= 1e-6)
      this[1] = 0;
    if (Math.abs(this[2]) <= 1e-6)
      this[2] = 0;
    return this;
  }
  negate() {
    this[0] = -this[0];
    this[1] = -this[1];
    this[2] = -this[2];
    return this;
  }
  clamp(min, max) {
    this[0] = Math.min(Math.max(this[0], min[0]), max[0]);
    this[1] = Math.min(Math.max(this[1], min[1]), max[1]);
    this[2] = Math.min(Math.max(this[2], min[2]), max[2]);
    return this;
  }
  dot(b) {
    return this[0] * b[0] + this[1] * b[1] + this[2] * b[2];
  }
  /** Align vector direction so its orthogonal to an axis direction */
  alignTwist(axis, dir) {
    this.fromCross(dir, axis).fromCross(axis, this);
    return this;
  }
  /** Shift current position to be on the plane */
  planeProj(planePos, planeNorm) {
    const planeConst = -Vec3.dot(planePos, planeNorm);
    const scl = -(Vec3.dot(planeNorm, this) + planeConst);
    this[0] += planeNorm[0] * scl;
    this[1] += planeNorm[1] * scl;
    this[2] += planeNorm[2] * scl;
    return this;
  }
  // #endregion
  // #region TRANFORMS
  transformQuat(q) {
    const qx = q[0], qy = q[1], qz = q[2], qw = q[3], vx = this[0], vy = this[1], vz = this[2], x1 = qy * vz - qz * vy, y1 = qz * vx - qx * vz, z1 = qx * vy - qy * vx, x2 = qw * x1 + qy * z1 - qz * y1, y2 = qw * y1 + qz * x1 - qx * z1, z2 = qw * z1 + qx * y1 - qy * x1;
    this[0] = vx + 2 * x2;
    this[1] = vy + 2 * y2;
    this[2] = vz + 2 * z2;
    return this;
  }
  axisAngle(axis, rad) {
    const cp = new Vec3().fromCross(axis, this), dot = Vec3.dot(axis, this), s = Math.sin(rad), c = Math.cos(rad), ci = 1 - c;
    this[0] = this[0] * c + cp[0] * s + axis[0] * dot * ci;
    this[1] = this[1] * c + cp[1] * s + axis[1] * dot * ci;
    this[2] = this[2] * c + cp[2] * s + axis[2] * dot * ci;
    return this;
  }
  rotate(rad, axis = "x") {
    const sin = Math.sin(rad), cos = Math.cos(rad), x = this[0], y = this[1], z = this[2];
    switch (axis) {
      case "y":
        this[0] = z * sin + x * cos;
        this[2] = z * cos - x * sin;
        break;
      case "x":
        this[1] = y * cos - z * sin;
        this[2] = y * sin + z * cos;
        break;
      case "z":
        this[0] = x * cos - y * sin;
        this[1] = x * sin + y * cos;
        break;
    }
    return this;
  }
  // #endregion
  // #region STATIC    
  static len(a) {
    return Math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2);
  }
  static lenSqr(a) {
    return a[0] ** 2 + a[1] ** 2 + a[2] ** 2;
  }
  static dist(a, b) {
    return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
  }
  static distSqr(a, b) {
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
  }
  static dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }
  static cross(a, b, out = new Vec3()) {
    const ax = a[0], ay = a[1], az = a[2], bx = b[0], by = b[1], bz = b[2];
    out[0] = ay * bz - az * by;
    out[1] = az * bx - ax * bz;
    out[2] = ax * by - ay * bx;
    return out;
  }
  static scaleThenAdd(scale, a, b, out = new Vec3()) {
    out[0] = a[0] * scale + b[0];
    out[1] = a[1] * scale + b[1];
    out[2] = a[2] * scale + b[2];
    return out;
  }
  static fromQuat(q, v = [0, 0, 1]) {
    return new Vec3(v).transformQuat(q);
  }
  static angle(a, b) {
    const d = this.dot(a, b), c = new Vec3().fromCross(a, b);
    return Math.atan2(Vec3.len(c), d);
  }
  /*
      static angleTo( from: ConstVec3, to: ConstVec3 ): number{
          // NOTE ORIG code doesn't work all the time
          // const denom = Math.sqrt( Vec3.lenSqr(from) * Vec3.lenSqr(to) );
          // if( denom < 0.00001 ) return 0;
          
          // const dot  = Math.min( 1, Math.max( -1, Vec3.dot( from, to ) / denom ));
          // const rad  = Math.acos( dot );
          // const sign = Math.sign( // Cross Product
          //     ( from[1] * to[2] - from[2] * to[1] ) + 
          //     ( from[2] * to[0] - from[0] * to[2] ) +
          //     ( from[0] * to[1] - from[1] * to[0] )
          // );
  
          const d    = Vec3.dot( from, to );
  
          console.log( 'dot', d );
  
          const c    = Vec3.cross( from, to );
          const rad  = Math.atan2( Vec3.len( c ), d );
          // c.norm();
          const sign = Math.sign( c[0] + c[1] + c[2] );// || 1;
          // const sign = Math.sign( to[0] * c[0] + to[1] * c[1] + to[2] * c[2] ) || 1;
          console.log( 'sign', sign );
          
          return rad * sign;
      }
      */
  /*
  static smoothDamp( cur: ConstVec3, tar: ConstVec3, vel: TVec3, dt: number, smoothTime: number = 0.25, maxSpeed: number = Infinity ): TVec3{
      // Based on Game Programming Gems 4 Chapter 1.10
      smoothTime   = Math.max( 0.0001, smoothTime );
      const omega  = 2 / smoothTime;
      const x      = omega * dt;
      const exp    = 1 / ( 1 + x + 0.48 * x * x + 0.235 * x * x * x );
  
      const change = vec3.sub( [0,0,0], cur, tar );
  
      // Clamp maximum speed
      const maxChange   = maxSpeed * smoothTime;
      const maxChangeSq = maxChange * maxChange;
      const magnitudeSq = change[0]**2 + change[1]**2 + change[2]**2;
  
      if( magnitudeSq > maxChangeSq ){
          const magnitude = Math.sqrt( magnitudeSq );
          vec3.scale( change, change, 1 / (magnitude * maxChange ) );
      }
  
      const diff = vec3.sub( [0,0,0], cur, change );
  
      // const tempX = ( velocity.x + omega * changeX ) * deltaTime;
      const temp  = vec3.scaleAndAdd( [0,0,0], vel, change, omega );
      vec3.scale( temp, temp, dt );
  
      // velocityR.x = ( velocity.x - omega * tempX ) * exp;
      vec3.scaleAndAdd( vel, vel, temp, -omega );
      vec3.scale( vel, vel, exp );
  
      // out.x = targetX + ( changeX + tempX ) * exp;
      const out = vec3.add( [0,0,0], change, temp );
      vec3.scale( out, out, exp );
      vec3.add( out, diff, out );
  
      // Prevent overshooting
      const origMinusCurrent = vec3.sub( [0,0,0], tar, cur );
      const outMinusOrig     = vec3.sub( [0,0,0], out, tar );
      if( origMinusCurrent[0] * outMinusOrig[0] + origMinusCurrent[1] * outMinusOrig[1] +  origMinusCurrent[2] * outMinusOrig[2] > -0.00001 ){
          vec3.copy( out, tar );
          vec3.copy( vel, [0,0,0] );
      }
  
      return out;
  }
  */
  // #endregion
}

class Quat extends Array {
  // #region STATIC CONSTANTS
  static LOOKXP = [0, -0.7071067811865475, 0, 0.7071067811865475];
  static LOOKXN = [0, 0.7071067811865475, 0, 0.7071067811865475];
  static LOOKYP = [0.7071067811865475, 0, 0, 0.7071067811865475];
  static LOOKYN = [-0.7071067811865475, 0, 0, 0.7071067811865475];
  static LOOKZP = [0, -1, 0, 0];
  static LOOKZN = [0, 0, 0, 1];
  // #endregion
  // #region MAIN
  constructor(v) {
    super(4);
    if (v instanceof Quat || v instanceof Float32Array || v instanceof Array && v.length == 4) {
      this[0] = v[0];
      this[1] = v[1];
      this[2] = v[2];
      this[3] = v[3];
    } else {
      this[0] = 0;
      this[1] = 0;
      this[2] = 0;
      this[3] = 1;
    }
  }
  // #endregion
  // #region SETTERS / GETTERS
  identity() {
    this[0] = 0;
    this[1] = 0;
    this[2] = 0;
    this[3] = 1;
    return this;
  }
  copy(a) {
    this[0] = a[0];
    this[1] = a[1];
    this[2] = a[2];
    this[3] = a[3];
    return this;
  }
  copyTo(a) {
    a[0] = this[0];
    a[1] = this[1];
    a[2] = this[2];
    a[3] = this[3];
    return this;
  }
  clone() {
    return new Quat(this);
  }
  // #endregion
  // #region FROM OPERATORS
  fromMul(a, b) {
    const ax = a[0], ay = a[1], az = a[2], aw = a[3], bx = b[0], by = b[1], bz = b[2], bw = b[3];
    this[0] = ax * bw + aw * bx + ay * bz - az * by;
    this[1] = ay * bw + aw * by + az * bx - ax * bz;
    this[2] = az * bw + aw * bz + ax * by - ay * bx;
    this[3] = aw * bw - ax * bx - ay * by - az * bz;
    return this;
  }
  /** Axis must be normlized, Angle in Radians  */
  fromAxisAngle(axis, rad) {
    const half = rad * 0.5;
    const s = Math.sin(half);
    this[0] = axis[0] * s;
    this[1] = axis[1] * s;
    this[2] = axis[2] * s;
    this[3] = Math.cos(half);
    return this;
  }
  /** Using unit vectors, Shortest swing rotation from Direction A to Direction B  */
  fromSwing(a, b) {
    const dot = Vec3.dot(a, b);
    if (dot < -0.999999) {
      const tmp = new Vec3().fromCross(Vec3.LEFT, a);
      if (tmp.len < 1e-6)
        tmp.fromCross(Vec3.UP, a);
      this.fromAxisAngle(tmp.norm(), Math.PI);
    } else if (dot > 0.999999) {
      this[0] = 0;
      this[1] = 0;
      this[2] = 0;
      this[3] = 1;
    } else {
      const v = Vec3.cross(a, b, [0, 0, 0]);
      this[0] = v[0];
      this[1] = v[1];
      this[2] = v[2];
      this[3] = 1 + dot;
      this.norm();
    }
    return this;
  }
  fromInvert(q) {
    const a0 = q[0], a1 = q[1], a2 = q[2], a3 = q[3], dot = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
    if (dot == 0) {
      this[0] = this[1] = this[2] = this[3] = 0;
      return this;
    }
    const invDot = 1 / dot;
    this[0] = -a0 * invDot;
    this[1] = -a1 * invDot;
    this[2] = -a2 * invDot;
    this[3] = a3 * invDot;
    return this;
  }
  fromNegate(q) {
    this[0] = -q[0];
    this[1] = -q[1];
    this[2] = -q[2];
    this[3] = -q[3];
    return this;
  }
  fromLookDir(dir, up = [0, 1, 0]) {
    const zAxis = new Vec3(dir).norm();
    const xAxis = new Vec3().fromCross(up, zAxis).norm();
    const yAxis = new Vec3().fromCross(zAxis, xAxis).norm();
    const m00 = xAxis[0], m01 = xAxis[1], m02 = xAxis[2], m10 = yAxis[0], m11 = yAxis[1], m12 = yAxis[2], m20 = zAxis[0], m21 = zAxis[1], m22 = zAxis[2], t = m00 + m11 + m22;
    let x, y, z, w, s;
    if (t > 0) {
      s = Math.sqrt(t + 1);
      w = s * 0.5;
      s = 0.5 / s;
      x = (m12 - m21) * s;
      y = (m20 - m02) * s;
      z = (m01 - m10) * s;
    } else if (m00 >= m11 && m00 >= m22) {
      s = Math.sqrt(1 + m00 - m11 - m22);
      x = 0.5 * s;
      s = 0.5 / s;
      y = (m01 + m10) * s;
      z = (m02 + m20) * s;
      w = (m12 - m21) * s;
    } else if (m11 > m22) {
      s = Math.sqrt(1 + m11 - m00 - m22);
      y = 0.5 * s;
      s = 0.5 / s;
      x = (m10 + m01) * s;
      z = (m21 + m12) * s;
      w = (m20 - m02) * s;
    } else {
      s = Math.sqrt(1 + m22 - m00 - m11);
      z = 0.5 * s;
      s = 0.5 / s;
      x = (m20 + m02) * s;
      y = (m21 + m12) * s;
      w = (m01 - m10) * s;
    }
    this[0] = x;
    this[1] = y;
    this[2] = z;
    this[3] = w;
    return this;
  }
  fromNBlend(a, b, t) {
    const a_x = a[0];
    const a_y = a[1];
    const a_z = a[2];
    const a_w = a[3];
    const b_x = b[0];
    const b_y = b[1];
    const b_z = b[2];
    const b_w = b[3];
    const dot = a_x * b_x + a_y * b_y + a_z * b_z + a_w * b_w;
    const ti = 1 - t;
    const s = dot < 0 ? -1 : 1;
    this[0] = ti * a_x + t * b_x * s;
    this[1] = ti * a_y + t * b_y * s;
    this[2] = ti * a_z + t * b_z * s;
    this[3] = ti * a_w + t * b_w * s;
    return this.norm();
  }
  /** Used to get data from a flat buffer */
  fromBuf(ary, idx) {
    this[0] = ary[idx];
    this[1] = ary[idx + 1];
    this[2] = ary[idx + 2];
    this[3] = ary[idx + 3];
    return this;
  }
  /** Put data into a flat buffer */
  toBuf(ary, idx) {
    ary[idx] = this[0];
    ary[idx + 1] = this[1];
    ary[idx + 2] = this[2];
    ary[idx + 3] = this[3];
    return this;
  }
  fromEuler(x, y, z) {
    let xx = 0, yy = 0, zz = 0;
    if (x instanceof Vec3 || x instanceof Float32Array || x instanceof Array && x.length == 3) {
      xx = x[0] * 0.01745329251 * 0.5;
      yy = x[1] * 0.01745329251 * 0.5;
      zz = x[2] * 0.01745329251 * 0.5;
    } else if (typeof x === "number" && typeof y === "number" && typeof z === "number") {
      xx = x * 0.01745329251 * 0.5;
      yy = y * 0.01745329251 * 0.5;
      zz = z * 0.01745329251 * 0.5;
    }
    const c1 = Math.cos(xx), c2 = Math.cos(yy), c3 = Math.cos(zz), s1 = Math.sin(xx), s2 = Math.sin(yy), s3 = Math.sin(zz);
    this[0] = s1 * c2 * c3 + c1 * s2 * s3;
    this[1] = c1 * s2 * c3 - s1 * c2 * s3;
    this[2] = c1 * c2 * s3 - s1 * s2 * c3;
    this[3] = c1 * c2 * c3 + s1 * s2 * s3;
    return this.norm();
  }
  // /** Create a rotation from eye & target position */
  // lookAt(
  //   out: TVec4,
  //   eye: TVec3, // Position of camera or object
  //   target: TVec3 = [0, 0, 0], // Position to look at
  //   up: TVec3 = [0, 1, 0], // Up direction for orientation
  // ): TVec4 {
  //   // Forward is inverted, will face correct direction when converted
  //   // to a ViewMatrix as it'll invert the Forward direction anyway
  //   const z: TVec3 = vec3.sub([0, 0, 0], eye, target);
  //   const x: TVec3 = vec3.cross([0, 0, 0], up, z);
  //   const y: TVec3 = vec3.cross([0, 0, 0], z, x);
  //   vec3.normalize(x, x);
  //   vec3.normalize(y, y);
  //   vec3.normalize(z, z);
  //   // Format: column-major, when typed out it looks like row-major
  //   quat.fromMat3(out, [...x, ...y, ...z]);
  //   return quat.normalize(out, out);
  // }
  // #endregion
  // #region OPERATORS
  /** Multiple Quaternion onto this Quaternion */
  mul(q) {
    const ax = this[0], ay = this[1], az = this[2], aw = this[3], bx = q[0], by = q[1], bz = q[2], bw = q[3];
    this[0] = ax * bw + aw * bx + ay * bz - az * by;
    this[1] = ay * bw + aw * by + az * bx - ax * bz;
    this[2] = az * bw + aw * bz + ax * by - ay * bx;
    this[3] = aw * bw - ax * bx - ay * by - az * bz;
    return this;
  }
  /** PreMultiple Quaternions onto this Quaternion */
  pmul(q) {
    const ax = q[0], ay = q[1], az = q[2], aw = q[3], bx = this[0], by = this[1], bz = this[2], bw = this[3];
    this[0] = ax * bw + aw * bx + ay * bz - az * by;
    this[1] = ay * bw + aw * by + az * bx - ax * bz;
    this[2] = az * bw + aw * bz + ax * by - ay * bx;
    this[3] = aw * bw - ax * bx - ay * by - az * bz;
    return this;
  }
  norm() {
    let len = this[0] ** 2 + this[1] ** 2 + this[2] ** 2 + this[3] ** 2;
    if (len > 0) {
      len = 1 / Math.sqrt(len);
      this[0] *= len;
      this[1] *= len;
      this[2] *= len;
      this[3] *= len;
    }
    return this;
  }
  invert() {
    const a0 = this[0], a1 = this[1], a2 = this[2], a3 = this[3], dot = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
    if (dot == 0) {
      this[0] = this[1] = this[2] = this[3] = 0;
      return this;
    }
    const invDot = 1 / dot;
    this[0] = -a0 * invDot;
    this[1] = -a1 * invDot;
    this[2] = -a2 * invDot;
    this[3] = a3 * invDot;
    return this;
  }
  negate() {
    this[0] = -this[0];
    this[1] = -this[1];
    this[2] = -this[2];
    this[3] = -this[3];
    return this;
  }
  // #endregion
  // #region ROTATIONS
  rotX(rad) {
    rad *= 0.5;
    const ax = this[0], ay = this[1], az = this[2], aw = this[3], bx = Math.sin(rad), bw = Math.cos(rad);
    this[0] = ax * bw + aw * bx;
    this[1] = ay * bw + az * bx;
    this[2] = az * bw - ay * bx;
    this[3] = aw * bw - ax * bx;
    return this;
  }
  rotY(rad) {
    rad *= 0.5;
    const ax = this[0], ay = this[1], az = this[2], aw = this[3], by = Math.sin(rad), bw = Math.cos(rad);
    this[0] = ax * bw - az * by;
    this[1] = ay * bw + aw * by;
    this[2] = az * bw + ax * by;
    this[3] = aw * bw - ay * by;
    return this;
  }
  rotZ(rad) {
    rad *= 0.5;
    const ax = this[0], ay = this[1], az = this[2], aw = this[3], bz = Math.sin(rad), bw = Math.cos(rad);
    this[0] = ax * bw + ay * bz;
    this[1] = ay * bw - ax * bz;
    this[2] = az * bw + aw * bz;
    this[3] = aw * bw - az * bz;
    return this;
  }
  rotDeg(deg, axis = 0) {
    const rad = deg * Math.PI / 180;
    switch (axis) {
      case 0:
        this.rotX(rad);
        break;
      case 1:
        this.rotY(rad);
        break;
      case 2:
        this.rotZ(rad);
        break;
    }
    return this;
  }
  // #endregion
  // #region SPECIAL OPERATORS
  /** Inverts the quaternion passed in, then pre multiplies to this quaternion. */
  pmulInvert(q) {
    let ax = q[0], ay = q[1], az = q[2], aw = q[3];
    const dot = ax * ax + ay * ay + az * az + aw * aw;
    if (dot === 0) {
      ax = ay = az = aw = 0;
    } else {
      const dot_inv = 1 / dot;
      ax = -ax * dot_inv;
      ay = -ay * dot_inv;
      az = -az * dot_inv;
      aw = aw * dot_inv;
    }
    const bx = this[0], by = this[1], bz = this[2], bw = this[3];
    this[0] = ax * bw + aw * bx + ay * bz - az * by;
    this[1] = ay * bw + aw * by + az * bx - ax * bz;
    this[2] = az * bw + aw * bz + ax * by - ay * bx;
    this[3] = aw * bw - ax * bx - ay * by - az * bz;
    return this;
  }
  pmulAxisAngle(axis, rad) {
    const half = rad * 0.5;
    const s = Math.sin(half);
    const ax = axis[0] * s;
    const ay = axis[1] * s;
    const az = axis[2] * s;
    const aw = Math.cos(half);
    const bx = this[0], by = this[1], bz = this[2], bw = this[3];
    this[0] = ax * bw + aw * bx + ay * bz - az * by;
    this[1] = ay * bw + aw * by + az * bx - ax * bz;
    this[2] = az * bw + aw * bz + ax * by - ay * bx;
    this[3] = aw * bw - ax * bx - ay * by - az * bz;
    return this;
  }
  dotNegate(q, chk) {
    if (Quat.dot(q, chk) < 0)
      this.fromNegate(q);
    else
      this.copy(q);
    return this;
  }
  // #endregion
  // #region STATIC
  static dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  }
  static lenSqr(a, b) {
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 + (a[3] - b[3]) ** 2;
  }
  static nblend(a, b, t, out) {
    const a_x = a[0];
    const a_y = a[1];
    const a_z = a[2];
    const a_w = a[3];
    const b_x = b[0];
    const b_y = b[1];
    const b_z = b[2];
    const b_w = b[3];
    const dot = a_x * b_x + a_y * b_y + a_z * b_z + a_w * b_w;
    const ti = 1 - t;
    const s = dot < 0 ? -1 : 1;
    out[0] = ti * a_x + t * b_x * s;
    out[1] = ti * a_y + t * b_y * s;
    out[2] = ti * a_z + t * b_z * s;
    out[3] = ti * a_w + t * b_w * s;
    return out.norm();
  }
  static slerp(a, b, t, out) {
    const ax = a[0], ay = a[1], az = a[2], aw = a[3];
    let bx = b[0], by = b[1], bz = b[2], bw = b[3];
    let omega, cosom, sinom, scale0, scale1;
    cosom = ax * bx + ay * by + az * bz + aw * bw;
    if (cosom < 0) {
      cosom = -cosom;
      bx = -bx;
      by = -by;
      bz = -bz;
      bw = -bw;
    }
    if (1 - cosom > 1e-6) {
      omega = Math.acos(cosom);
      sinom = Math.sin(omega);
      scale0 = Math.sin((1 - t) * omega) / sinom;
      scale1 = Math.sin(t * omega) / sinom;
    } else {
      scale0 = 1 - t;
      scale1 = t;
    }
    out[0] = scale0 * ax + scale1 * bx;
    out[1] = scale0 * ay + scale1 * by;
    out[2] = scale0 * az + scale1 * bz;
    out[3] = scale0 * aw + scale1 * bw;
    return out;
  }
  static shortest(from, to, out) {
    const ax = from[0], ay = from[1], az = from[2], aw = from[3];
    let bx = to[0], by = to[1], bz = to[2], bw = to[3];
    const dot = ax * bx + ay * by + az * bz + aw * bw;
    if (dot < 0) {
      bx = -bx;
      by = -by;
      bz = -bz;
      bw = -bw;
    }
    const d = bx * bx + by * by + bz * bz + bw * bw;
    if (d === 0) {
      bx = 0;
      by = 0;
      bz = 0;
      bw = 0;
    } else {
      const di = 1 / d;
      bx = -bx * di;
      by = -by * di;
      bz = -bz * di;
      bw = bw * di;
    }
    out[0] = ax * bw + aw * bx + ay * bz - az * by;
    out[1] = ay * bw + aw * by + az * bx - ax * bz;
    out[2] = az * bw + aw * bz + ax * by - ay * bx;
    out[3] = aw * bw - ax * bx - ay * by - az * bz;
    return out;
  }
  static swing(a, b) {
    return new Quat().fromSwing(a, b);
  }
  static axisAngle(axis, rad) {
    return new Quat().fromAxisAngle(axis, rad);
  }
  // // https://pastebin.com/66qSCKcZ
  // // https://forum.unity.com/threads/manually-calculate-angular-velocity-of-gameobject.289462/#post-4302796
  // static angularVelocity( foreLastFrameRotation: ConstQuat, lastFrameRotation: ConstQuat): TVec3{
  //     var q = lastFrameRotation * Quaternion.Inverse(foreLastFrameRotation);
  //     // no rotation?
  //     // You may want to increase this closer to 1 if you want to handle very small rotations.
  //     // Beware, if it is too close to one your answer will be Nan
  //     if ( Mathf.Abs(q.w) > 1023.5f / 1024.0f ) return [0,0,0]; Vector3.zero;
  //     float gain;
  //     // handle negatives, we could just flip it but this is faster
  //     if( q.w < 0.0f ){
  //         var angle = Mathf.Acos(-q.w);
  //         gain = -2.0f * angle / (Mathf.Sin(angle) * Time.deltaTime);
  //     }else{
  //         var angle = Mathf.Acos(q.w);
  //         gain = 2.0f * angle / (Mathf.Sin(angle) * Time.deltaTime);
  //     }
  //     Vector3 angularVelocity = new Vector3(q.x * gain, q.y * gain, q.z * gain);
  //     if(float.IsNaN(angularVelocity.z)) angularVelocity = Vector3.zero;
  //     return angularVelocity;
  // }
  // #endregion
}

class Transform {
  // #region MAIN
  rot = new Quat();
  pos = new Vec3(0);
  scl = new Vec3(1);
  constructor(rot, pos, scl) {
    if (rot instanceof Transform) {
      this.copy(rot);
    } else if (rot && pos && scl) {
      this.set(rot, pos, scl);
    }
  }
  // #endregion
  // #region SETTERS / GETTERS
  reset() {
    this.rot.identity();
    this.pos.xyz(0, 0, 0);
    this.scl.xyz(1, 1, 1);
    return this;
  }
  copy(t) {
    this.rot.copy(t.rot);
    this.pos.copy(t.pos);
    this.scl.copy(t.scl);
    return this;
  }
  set(r, p, s) {
    if (r)
      this.rot.copy(r);
    if (p)
      this.pos.copy(p);
    if (s)
      this.scl.copy(s);
    return this;
  }
  clone() {
    return new Transform(this);
  }
  mul(cr, cp, cs) {
    if (cr instanceof Transform) {
      cp = cr.pos;
      cs = cr.scl;
      cr = cr.rot;
    }
    if (cr && cp) {
      this.pos.add(new Vec3().fromMul(this.scl, cp).transformQuat(this.rot));
      if (cs)
        this.scl.mul(cs);
      this.rot.mul(cr);
    }
    return this;
  }
  pmul(pr, pp, ps) {
    if (pr instanceof Transform) {
      pp = pr.pos;
      ps = pr.scl;
      pr = pr.rot;
    }
    if (!pr || !pp || !ps)
      return this;
    this.pos.mul(ps).transformQuat(pr).add(pp);
    if (ps)
      this.scl.mul(ps);
    this.rot.pmul(pr);
    return this;
  }
  addPos(cp, ignoreScl = false) {
    if (ignoreScl)
      this.pos.add(new Vec3().fromQuat(this.rot, cp));
    else
      this.pos.add(new Vec3().fromMul(cp, this.scl).transformQuat(this.rot));
    return this;
  }
  // #endregion
  // #region FROM OPERATORS
  fromMul(tp, tc) {
    const v = new Vec3().fromMul(tp.scl, tc.pos).transformQuat(tp.rot);
    this.pos.fromAdd(tp.pos, v);
    this.scl.fromMul(tp.scl, tc.scl);
    this.rot.fromMul(tp.rot, tc.rot);
    return this;
  }
  fromInvert(t) {
    this.rot.fromInvert(t.rot);
    this.scl.fromInvert(t.scl);
    this.pos.fromNegate(t.pos).mul(this.scl).transformQuat(this.rot);
    return this;
  }
  // #endregion
  // #region TRANSFORMATION
  transformVec3(v, out) {
    return (out || v).fromMul(v, this.scl).transformQuat(this.rot).add(this.pos);
  }
  // #endregion
}

class Bone {
  // #region MAIN
  index = -1;
  // Array Index
  pindex = -1;
  // Array Index of Parent
  name = "";
  // Bone Name
  len = 0;
  // Length of Bone
  local = new Transform();
  // Local space transform
  world = new Transform();
  // World space transform
  constraint = null;
  constructor(props) {
    this.name = props?.name ? props.name : "bone" + Math.random();
    if (typeof props?.parent === "number")
      this.pindex = props.parent;
    if (props?.parent instanceof Bone)
      this.pindex = props.parent.index;
    if (props?.rot)
      this.local.rot.copy(props.rot);
    if (props?.pos)
      this.local.pos.copy(props.pos);
    if (props?.scl)
      this.local.scl.copy(props.scl);
    if (props?.len)
      this.len = props.len;
  }
  // #endregion
  // #region METHODS
  clone() {
    const b = new Bone();
    b.name = this.name;
    b.index = this.index;
    b.pindex = this.pindex;
    b.len = this.len;
    b.constraint = this.constraint;
    b.local.copy(this.local);
    b.world.copy(this.world);
    return b;
  }
  // #endregion
}

class Pose {
  // #region MAIN
  arm;
  offset = new Transform();
  // Additional offset transformation to apply to pose root
  linkedBone = void 0;
  // This skeleton extends another skeleton
  bones = new Array();
  // Bone transformation heirarchy
  constructor(arm) {
    if (arm)
      this.arm = arm;
    if (arm?.poses?.bind) {
      for (let i = 0; i < arm.poses.bind.bones.length; i++) {
        this.bones.push(arm.poses.bind.bones[i].clone());
      }
      this.offset.copy(arm.poses.bind.offset);
    }
  }
  // #endregion
  // #region GETTERS
  getBone(o) {
    switch (typeof o) {
      case "string": {
        const idx = this.arm.names.get(o);
        return idx !== void 0 ? this.bones[idx] : null;
      }
      case "number":
        return this.bones[o];
    }
    return null;
  }
  getBones(ary) {
    const rtn = [];
    let b;
    for (const i of ary) {
      if (b = this.getBone(i))
        rtn.push(b);
    }
    return rtn;
  }
  clone() {
    const p = new Pose();
    p.arm = this.arm;
    p.offset.copy(this.offset);
    for (const b of this.bones)
      p.bones.push(b.clone());
    return p;
  }
  getWorldTailPos(o, out = new Vec3()) {
    const b = this.getBone(o);
    if (b)
      b.world.transformVec3(out.xyz(0, b.len, 0));
    return out;
  }
  // #endregion
  // #region SETTERS
  setLocalPos(boneId, v) {
    const bone = this.getBone(boneId);
    if (bone)
      bone.local.pos.copy(v);
    return this;
  }
  setLocalRot(boneId, v) {
    const bone = this.getBone(boneId);
    if (bone)
      bone.local.rot.copy(v);
    return this;
  }
  copy(pose) {
    const bLen = this.bones.length;
    for (let i = 0; i < bLen; i++) {
      this.bones[i].local.copy(pose.bones[i].local);
      this.bones[i].world.copy(pose.bones[i].world);
    }
    return this;
  }
  // #endregion
  // #region COMPUTE
  updateWorld() {
    for (const b of this.bones) {
      if (b.pindex !== -1) {
        b.world.fromMul(this.bones[b.pindex].world, b.local);
      } else {
        b.world.fromMul(this.offset, b.local);
        if (this.linkedBone) {
          b.world.pmul(this.linkedBone.world);
        }
      }
    }
    return this;
  }
  updateWorldChildren(pIdx, incParent = false) {
    const parents = [pIdx];
    let b;
    if (incParent) {
      b = this.bones[pIdx];
      b.world.fromMul(
        b.pindex !== -1 ? this.bones[b.pindex].world : this.offset,
        b.local
      );
    }
    for (let i = pIdx + 1; i < this.bones.length; i++) {
      b = this.bones[i];
      if (parents.indexOf(b.pindex) === -1)
        continue;
      b.world.fromMul(this.bones[b.pindex].world, b.local);
      parents.push(b.index);
    }
    return this;
  }
  // updateLocalRot(): this{
  //     let b;
  //     for( b of this.bones ){
  //         b.local.rot
  //             .copy( b.world.rot )
  //             .pmulInvert( \
  //                 ( b.pindex !== -1 )?
  //                     this.bones[ b.pindex ].world.rot :
  //                     this.offset.rot
  //             );
  //     }
  //     return this;
  // }
  getWorldRotation(boneId, out = new Quat()) {
    let bone = this.getBone(boneId);
    if (!bone) {
      if (boneId === -1)
        out.copy(this.offset.rot);
      else
        console.error("Pose.getWorldRotation - bone not found", boneId);
      return out;
    }
    out.copy(bone.local.rot);
    while (bone.pindex !== -1) {
      bone = this.bones[bone.pindex];
      out.pmul(bone.local.rot);
    }
    out.pmul(this.offset.rot);
    if (this.linkedBone)
      out.pmul(this.linkedBone.world.rot);
    return out;
  }
  getWorldTransform(boneId, out = new Transform()) {
    let bone = this.getBone(boneId);
    if (!bone) {
      if (boneId === -1)
        out.copy(this.offset);
      else
        console.error("Pose.getWorldTransform - bone not found", boneId);
      return out;
    }
    out.copy(bone.local);
    while (bone.pindex !== -1) {
      bone = this.bones[bone.pindex];
      out.pmul(bone.local);
    }
    out.pmul(this.offset);
    if (this.linkedBone)
      out.pmul(this.linkedBone.world);
    return out;
  }
  getWorldPosition(boneId, out = new Vec3()) {
    return out.copy(this.getWorldTransform(boneId).pos);
  }
  // #endregion
  // #region OPERATIONS
  rotLocal(boneId, deg, axis = 0) {
    const bone = this.getBone(boneId);
    if (bone) {
      switch (axis) {
        case 1:
          bone.local.rot.rotY(deg * Math.PI / 180);
          break;
        case 2:
          bone.local.rot.rotZ(deg * Math.PI / 180);
          break;
        default:
          bone.local.rot.rotX(deg * Math.PI / 180);
          break;
      }
    } else
      console.warn("Bone not found, ", boneId);
    return this;
  }
  rotWorld(boneId, deg, axis = "x") {
    const bone = this.getBone(boneId);
    if (bone) {
      const pWRot = this.getWorldRotation(bone.pindex);
      const cWRot = new Quat(pWRot).mul(bone.local.rot);
      const ax = axis == "y" ? [0, 1, 0] : axis == "z" ? [0, 0, 1] : [1, 0, 0];
      cWRot.pmulAxisAngle(ax, deg * Math.PI / 180).pmulInvert(pWRot).copyTo(bone.local.rot);
    } else {
      console.error("Pose.rotWorld - bone not found", boneId);
    }
    return this;
  }
  moveLocal(boneId, offset) {
    const bone = this.getBone(boneId);
    if (bone)
      bone.local.pos.add(offset);
    else
      console.warn("Pose.moveLocal - bone not found, ", boneId);
    return this;
  }
  posLocal(boneId, pos) {
    const bone = this.getBone(boneId);
    if (bone)
      bone.local.pos.copy(pos);
    else
      console.warn("Pose.posLocal - bone not found, ", boneId);
    return this;
  }
  sclLocal(boneId, v) {
    const bone = this.getBone(boneId);
    if (bone) {
      if (v instanceof Array || v instanceof Float32Array)
        bone.local.scl.copy(v);
      else if (typeof v === "number")
        bone.local.scl.xyz(v, v, v);
    } else {
      console.warn("Pose.sclLocal - bone not found, ", boneId);
    }
    return this;
  }
  // #endregion
}

class Armature {
  // #region MAIN
  skin;
  names = /* @__PURE__ */ new Map();
  poses = {
    bind: new Pose(this)
  };
  // #endregion
  // #region GETTERS
  get bindPose() {
    return this.poses.bind;
  }
  get boneCount() {
    return this.poses.bind.bones.length;
  }
  // TODO: Maybe a better way for quick way to creates poses other then new Pose( Armature );
  newPose(saveAs) {
    const p = this.poses.bind.clone();
    if (saveAs)
      this.poses[saveAs] = p;
    return p;
  }
  // #endregion
  // #region METHODS
  addBone(obj) {
    const bones = this.poses.bind.bones;
    const idx = bones.length;
    if (obj instanceof Bone) {
      obj.index = idx;
      bones.push(obj);
      this.names.set(obj.name, idx);
      return obj;
    } else {
      const bone = new Bone(obj);
      bone.index = idx;
      bones.push(bone);
      this.names.set(bone.name, idx);
      if (typeof obj?.parent === "string") {
        const pIdx = this.names.get(obj?.parent);
        if (pIdx !== void 0)
          bone.pindex = pIdx;
        else
          console.error("Parent bone not found", obj.name);
      }
      return bone;
    }
  }
  getBone(o) {
    switch (typeof o) {
      case "string": {
        const idx = this.names.get(o);
        return idx !== void 0 ? this.poses.bind.bones[idx] : null;
      }
      case "number":
        return this.poses.bind.bones[o];
    }
    return null;
  }
  getBones(ary) {
    const rtn = [];
    let b;
    for (const i of ary) {
      if (b = this.getBone(i))
        rtn.push(b);
    }
    return rtn;
  }
  bind(boneLen = 0.2) {
    this.poses.bind.updateWorld();
    this.updateBoneLengths(this.poses.bind, boneLen);
    return this;
  }
  // Valid useage
  // const skin = arm.useSkin( new MatrixSkin( arm.bindPose ) );
  // const skin = arm.useSkin( MatrixSkin );
  useSkin(skin) {
    switch (typeof skin) {
      case "object":
        this.skin = skin;
        break;
      case "function":
        this.skin = new skin(this.poses.bind);
        break;
      default:
        console.error("Armature.useSkin, unknown typeof of skin ref", skin);
        break;
    }
    return this.skin;
  }
  // #endregion
  // #region #COMPUTE
  updateBoneLengths(_pose, boneLen = 0.1) {
    const pose = _pose || this.poses.bind;
    const bEnd = pose.bones.length - 1;
    let b;
    let p;
    for (let i = bEnd; i >= 0; i--) {
      b = pose.bones[i];
      if (b.pindex !== -1) {
        p = pose.bones[b.pindex];
        p.len = Vec3.dist(p.world.pos, b.world.pos);
        if (p.len < 1e-4)
          p.len = 0;
      }
    }
    if (boneLen != 0) {
      for (b of pose.bones) {
        if (b.len == 0)
          b.len = boneLen;
      }
    }
    return this;
  }
  // #endregion
}

class BoneBindings {
  // #region MAIN
  onUpdate;
  items = /* @__PURE__ */ new Map();
  constructor(fn) {
    this.onUpdate = fn;
  }
  // #endregion
  // #region METHODS
  bind(bone, obj) {
    this.items.set(window.crypto.randomUUID(), {
      bone: new WeakRef(bone),
      obj: new WeakRef(obj)
    });
    return this;
  }
  removeBone(bone) {
    const trash = [];
    let b;
    let k;
    let v;
    for ([k, v] of this.items) {
      b = v.bone.deref();
      if (!b || b === bone)
        trash.push(k);
    }
    if (trash.length > 0) {
      for (k of trash)
        this.items.delete(k);
    }
    return this;
  }
  updateAll() {
    const trash = [];
    let b;
    let o;
    let k;
    let v;
    for ([k, v] of this.items) {
      b = v.bone.deref();
      o = v.obj.deref();
      if (b && o)
        this.onUpdate(b, o);
      else
        trash.push(k);
    }
    if (trash.length > 0) {
      for (k of trash)
        this.items.delete(k);
    }
    return this;
  }
  // #endregion
}

class SocketItem {
  local = new Transform();
  obj;
  constructor(obj, pos, rot, scl) {
    this.obj = obj;
    if (pos)
      this.local.pos.copy(pos);
    if (rot)
      this.local.rot.copy(rot);
    if (scl)
      this.local.scl.copy(scl);
  }
}
class Socket {
  boneIndex = -1;
  local = new Transform();
  items = [];
  constructor(bi, pos, rot) {
    this.boneIndex = bi;
    if (pos)
      this.local.pos.copy(pos);
    if (rot)
      this.local.rot.copy(rot);
  }
}
class BoneSockets {
  sockets = /* @__PURE__ */ new Map();
  transformHandler;
  constructor(tHandler) {
    if (tHandler)
      this.transformHandler = tHandler;
  }
  add(name, bone, pos, rot) {
    this.sockets.set(name, new Socket(bone.index, pos, rot));
    return this;
  }
  attach(socketName, obj, pos, rot, scl) {
    const s = this.sockets.get(socketName);
    if (s)
      s.items.push(new SocketItem(obj, pos, rot, scl));
    else
      console.error("Socket.attach: Socket name not found:", socketName);
    return this;
  }
  updateFromPose(pose) {
    if (!this.transformHandler)
      return;
    const st = new Transform();
    const t = new Transform();
    let b;
    for (const s of this.sockets.values()) {
      b = pose.bones[s.boneIndex];
      st.fromMul(b.world, s.local);
      for (const i of s.items) {
        t.fromMul(st, i.local);
        try {
          this.transformHandler(t, i.obj);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          console.error("Error updating bone socket item: ", msg);
        }
      }
    }
  }
  debugAll(pose, debug) {
    const t = new Transform();
    let b;
    for (const s of this.sockets.values()) {
      b = pose.bones[s.boneIndex];
      t.fromMul(b.world, s.local);
      debug.pnt.add(t.pos, 16777215, 4, 2);
    }
  }
}

class BoneMap {
  bones = /* @__PURE__ */ new Map();
  obj;
  constructor(obj) {
    if (obj)
      this.from(obj);
  }
  from(obj) {
    this.obj = obj;
    const bAry = obj instanceof Armature ? obj.bindPose.bones : obj.bones;
    let bp;
    let bi;
    let key;
    for (const b of bAry) {
      for (bp of Parsers) {
        if (!(key = bp.test(b.name)))
          continue;
        bi = this.bones.get(key);
        if (!bi)
          this.bones.set(key, new BoneInfo(b));
        else if (bi && bp.isChain)
          bi.push(b);
        break;
      }
    }
  }
  getBoneMap(name) {
    return this.bones.get(name);
  }
  getBoneIndex(name) {
    const bi = this.bones.get(name);
    return bi ? bi.items[0].index : -1;
  }
  getBones(aryNames) {
    const bAry = this.obj instanceof Armature ? this.obj.bindPose.bones : this.obj.bones;
    const rtn = [];
    let bi;
    let i;
    for (const name of aryNames) {
      bi = this.bones.get(name);
      if (bi) {
        for (i of bi.items)
          rtn.push(bAry[i.index]);
      } else {
        console.warn("Bonemap.getBones - Bone not found", name);
      }
    }
    return rtn.length >= aryNames.length ? rtn : null;
  }
  getBoneNames(ary) {
    const rtn = [];
    let bi;
    let i;
    for (const name of ary) {
      if (bi = this.bones.get(name)) {
        for (i of bi.items)
          rtn.push(i.name);
      } else {
        console.warn("Bonemap.getBoneNames - Bone not found", name);
        return null;
      }
    }
    return rtn;
  }
  getChestBone() {
    const bAry = this.obj instanceof Armature ? this.obj.bindPose.bones : this.obj.bones;
    const rtn = [];
    const bi = this.bones.get("spine");
    if (bi) {
      rtn.push(bAry[bi.lastIndex]);
    }
    return rtn.length > 0 ? rtn : null;
  }
}
class BoneInfo {
  items = [];
  constructor(b) {
    if (b)
      this.push(b);
  }
  push(bone) {
    this.items.push({ index: bone.index, name: bone.name });
    return this;
  }
  get isChain() {
    return this.items.length > 1;
  }
  get count() {
    return this.items.length;
  }
  get index() {
    return this.items[0].index;
  }
  get lastIndex() {
    return this.items[this.items.length - 1].index;
  }
}
class BoneParse {
  name;
  isLR;
  isChain;
  reFind;
  reExclude;
  constructor(name, isLR, reFind, reExclude, isChain = false) {
    this.name = name;
    this.isLR = isLR;
    this.isChain = isChain;
    this.reFind = new RegExp(reFind, "i");
    if (reExclude)
      this.reExclude = new RegExp(reExclude, "i");
  }
  test(bname) {
    if (!this.reFind.test(bname))
      return null;
    if (this.reExclude && this.reExclude.test(bname))
      return null;
    if (this.isLR && reLeft.test(bname))
      return this.name + "_l";
    if (this.isLR && reRight.test(bname))
      return this.name + "_r";
    return this.name;
  }
}
const reLeft = new RegExp("\\.l|left|_l", "i");
const reRight = new RegExp("\\.r|right|_r", "i");
const Parsers = [
  new BoneParse("thigh", true, "thigh|up.*leg", "twist"),
  //upleg | upperleg
  new BoneParse("shin", true, "shin|leg|calf", "up|twist"),
  new BoneParse("foot", true, "foot"),
  new BoneParse("toe", true, "toe"),
  new BoneParse("shoulder", true, "clavicle|shoulder"),
  new BoneParse("upperarm", true, "(upper.*arm|arm)", "fore|twist|lower"),
  new BoneParse("forearm", true, "forearm|arm", "up|twist"),
  new BoneParse("hand", true, "hand", "thumb|index|middle|ring|pinky"),
  new BoneParse("head", false, "head"),
  new BoneParse("neck", false, "neck"),
  new BoneParse("hip", false, "hips*|pelvis"),
  new BoneParse("root", false, "root"),
  // eslint-disable-next-line no-useless-escape
  new BoneParse("spine", false, "spine.*d*|chest", void 0, true)
];

class Mat4 extends Array {
  // #region STATIC VALUES
  static BYTESIZE = 16 * 4;
  // #endregion
  // #region CONSTRUCTOR
  constructor(v) {
    super(16);
    if (v)
      this.copy(v);
    else {
      this[0] = 1;
      this[1] = 0;
      this[2] = 0;
      this[3] = 0;
      this[4] = 0;
      this[5] = 1;
      this[6] = 0;
      this[7] = 0;
      this[8] = 0;
      this[9] = 0;
      this[10] = 1;
      this[11] = 0;
      this[12] = 0;
      this[13] = 0;
      this[14] = 0;
      this[15] = 1;
    }
  }
  // #endregion
  // #region GETTERS / SETTERS
  identity() {
    this[0] = 1;
    this[1] = 0;
    this[2] = 0;
    this[3] = 0;
    this[4] = 0;
    this[5] = 1;
    this[6] = 0;
    this[7] = 0;
    this[8] = 0;
    this[9] = 0;
    this[10] = 1;
    this[11] = 0;
    this[12] = 0;
    this[13] = 0;
    this[14] = 0;
    this[15] = 1;
    return this;
  }
  clearTranslation() {
    this[12] = this[13] = this[14] = 0;
    this[15] = 1;
    return this;
  }
  // copy another matrix's data to this one.
  copy(mat, offset = 0) {
    let i;
    for (i = 0; i < 16; i++)
      this[i] = mat[offset + i];
    return this;
  }
  copyTo(out) {
    let i;
    for (i = 0; i < 16; i++)
      out[i] = this[i];
    return this;
  }
  determinant() {
    const a00 = this[0], a01 = this[1], a02 = this[2], a03 = this[3], a10 = this[4], a11 = this[5], a12 = this[6], a13 = this[7], a20 = this[8], a21 = this[9], a22 = this[10], a23 = this[11], a30 = this[12], a31 = this[13], a32 = this[14], a33 = this[15], b0 = a00 * a11 - a01 * a10, b1 = a00 * a12 - a02 * a10, b2 = a01 * a12 - a02 * a11, b3 = a20 * a31 - a21 * a30, b4 = a20 * a32 - a22 * a30, b5 = a21 * a32 - a22 * a31, b6 = a00 * b5 - a01 * b4 + a02 * b3, b7 = a10 * b5 - a11 * b4 + a12 * b3, b8 = a20 * b2 - a21 * b1 + a22 * b0, b9 = a30 * b2 - a31 * b1 + a32 * b0;
    return a13 * b6 - a03 * b7 + a33 * b8 - a23 * b9;
  }
  /** Frobenius norm of a Matrix */
  frob() {
    return Math.hypot(
      this[0],
      this[1],
      this[2],
      this[3],
      this[4],
      this[5],
      this[6],
      this[7],
      this[8],
      this[9],
      this[10],
      this[11],
      this[12],
      this[13],
      this[14],
      this[15]
    );
  }
  //----------------------------------------------------
  getTranslation(out) {
    out = out || [0, 0, 0];
    out[0] = this[12];
    out[1] = this[13];
    out[2] = this[14];
    return out;
  }
  getScale(out) {
    const m11 = this[0], m12 = this[1], m13 = this[2], m21 = this[4], m22 = this[5], m23 = this[6], m31 = this[8], m32 = this[9], m33 = this[10];
    out = out || [0, 0, 0];
    out[0] = Math.sqrt(m11 * m11 + m12 * m12 + m13 * m13);
    out[1] = Math.sqrt(m21 * m21 + m22 * m22 + m23 * m23);
    out[2] = Math.sqrt(m31 * m31 + m32 * m32 + m33 * m33);
    return out;
  }
  getRotation(out) {
    const trace = this[0] + this[5] + this[10];
    let S = 0;
    out = out || [0, 0, 0, 1];
    if (trace > 0) {
      S = Math.sqrt(trace + 1) * 2;
      out[3] = 0.25 * S;
      out[0] = (this[6] - this[9]) / S;
      out[1] = (this[8] - this[2]) / S;
      out[2] = (this[1] - this[4]) / S;
    } else if (this[0] > this[5] && this[0] > this[10]) {
      S = Math.sqrt(1 + this[0] - this[5] - this[10]) * 2;
      out[3] = (this[6] - this[9]) / S;
      out[0] = 0.25 * S;
      out[1] = (this[1] + this[4]) / S;
      out[2] = (this[8] + this[2]) / S;
    } else if (this[5] > this[10]) {
      S = Math.sqrt(1 + this[5] - this[0] - this[10]) * 2;
      out[3] = (this[8] - this[2]) / S;
      out[0] = (this[1] + this[4]) / S;
      out[1] = 0.25 * S;
      out[2] = (this[6] + this[9]) / S;
    } else {
      S = Math.sqrt(1 + this[10] - this[0] - this[5]) * 2;
      out[3] = (this[1] - this[4]) / S;
      out[0] = (this[8] + this[2]) / S;
      out[1] = (this[6] + this[9]) / S;
      out[2] = 0.25 * S;
    }
    return out;
  }
  //----------------------------------------------------
  fromPerspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy * 0.5), nf = 1 / (near - far);
    this[0] = f / aspect;
    this[1] = 0;
    this[2] = 0;
    this[3] = 0;
    this[4] = 0;
    this[5] = f;
    this[6] = 0;
    this[7] = 0;
    this[8] = 0;
    this[9] = 0;
    this[10] = (far + near) * nf;
    this[11] = -1;
    this[12] = 0;
    this[13] = 0;
    this[14] = 2 * far * near * nf;
    this[15] = 0;
    return this;
  }
  /*
      Generates a perspective projection matrix with the given bounds.
      * The near/far clip planes correspond to a normalized device coordinate Z range of [-1, 1],
      export function perspectiveNO(out, fovy, aspect, near, far) {
      const f = 1.0 / Math.tan(fovy / 2);
      out[0] = f / aspect;
      out[1] = 0;
      out[2] = 0;
      out[3] = 0;
      out[4] = 0;
      out[5] = f;
      out[6] = 0;
      out[7] = 0;
      out[8] = 0;
      out[9] = 0;
      out[11] = -1;
      out[12] = 0;
      out[13] = 0;
      out[15] = 0;
      if (far != null && far !== Infinity) {
          const nf = 1 / (near - far);
          out[10] = (far + near) * nf;
          out[14] = 2 * far * near * nf;
      } else {
          out[10] = -1;
          out[14] = -2 * near;
      }
      return out;
      }
  
      Generates a perspective projection matrix suitable for WebGPU with the given bounds.
      The near/far clip planes correspond to a normalized device coordinate Z range of [0, 1],
      export function perspectiveZO(out, fovy, aspect, near, far) {
          const f = 1.0 / Math.tan(fovy / 2);
          out[0] = f / aspect;
          out[1] = 0;
          out[2] = 0;
          out[3] = 0;
          out[4] = 0;
          out[5] = f;
          out[6] = 0;
          out[7] = 0;
          out[8] = 0;
          out[9] = 0;
          out[11] = -1;
          out[12] = 0;
          out[13] = 0;
          out[15] = 0;
          if (far != null && far !== Infinity) {
            const nf = 1 / (near - far);
            out[10] = far * nf;
            out[14] = far * near * nf;
          } else {
            out[10] = -1;
            out[14] = -near;
          }
          return out;
        }
  
       * Generates a perspective projection matrix with the given field of view.
      * This is primarily useful for generating projection matrices to be used
      * with the still experiemental WebVR API.
      export function perspectiveFromFieldOfView(out, fov, near, far) {
          let upTan = Math.tan((fov.upDegrees * Math.PI) / 180.0);
          let downTan = Math.tan((fov.downDegrees * Math.PI) / 180.0);
          let leftTan = Math.tan((fov.leftDegrees * Math.PI) / 180.0);
          let rightTan = Math.tan((fov.rightDegrees * Math.PI) / 180.0);
          let xScale = 2.0 / (leftTan + rightTan);
          let yScale = 2.0 / (upTan + downTan);
      
          out[0] = xScale;
          out[1] = 0.0;
          out[2] = 0.0;
          out[3] = 0.0;
          out[4] = 0.0;
          out[5] = yScale;
          out[6] = 0.0;
          out[7] = 0.0;
          out[8] = -((leftTan - rightTan) * xScale * 0.5);
          out[9] = (upTan - downTan) * yScale * 0.5;
          out[10] = far / (near - far);
          out[11] = -1.0;
          out[12] = 0.0;
          out[13] = 0.0;
          out[14] = (far * near) / (near - far);
          out[15] = 0.0;
          return out;
      }
  
      */
  fromOrtho(left, right, bottom, top, near, far) {
    const lr = 1 / (left - right), bt = 1 / (bottom - top), nf = 1 / (near - far);
    this[0] = -2 * lr;
    this[1] = 0;
    this[2] = 0;
    this[3] = 0;
    this[4] = 0;
    this[5] = -2 * bt;
    this[6] = 0;
    this[7] = 0;
    this[8] = 0;
    this[9] = 0;
    this[10] = 2 * nf;
    this[11] = 0;
    this[12] = (left + right) * lr;
    this[13] = (top + bottom) * bt;
    this[14] = (far + near) * nf;
    this[15] = 1;
    return this;
  }
  /*
      * Generates a orthogonal projection matrix with the given bounds.
      * The near/far clip planes correspond to a normalized device coordinate Z range of [-1, 1],
      * which matches WebGL/OpenGL's clip volume.
     export function orthoNO(out, left, right, bottom, top, near, far) {
       const lr = 1 / (left - right);
       const bt = 1 / (bottom - top);
       const nf = 1 / (near - far);
       out[0] = -2 * lr;
       out[1] = 0;
       out[2] = 0;
       out[3] = 0;
       out[4] = 0;
       out[5] = -2 * bt;
       out[6] = 0;
       out[7] = 0;
       out[8] = 0;
       out[9] = 0;
       out[10] = 2 * nf;
       out[11] = 0;
       out[12] = (left + right) * lr;
       out[13] = (top + bottom) * bt;
       out[14] = (far + near) * nf;
       out[15] = 1;
       return out;
     }
  
      * Generates a orthogonal projection matrix with the given bounds.
      * The near/far clip planes correspond to a normalized device coordinate Z range of [0, 1],
      * which matches WebGPU/Vulkan/DirectX/Metal's clip volume.
      export function orthoZO(out, left, right, bottom, top, near, far) {
      const lr = 1 / (left - right);
      const bt = 1 / (bottom - top);
      const nf = 1 / (near - far);
      out[0] = -2 * lr;
      out[1] = 0;
      out[2] = 0;
      out[3] = 0;
      out[4] = 0;
      out[5] = -2 * bt;
      out[6] = 0;
      out[7] = 0;
      out[8] = 0;
      out[9] = 0;
      out[10] = nf;
      out[11] = 0;
      out[12] = (left + right) * lr;
      out[13] = (top + bottom) * bt;
      out[14] = near * nf;
      out[15] = 1;
      return out;
      }
  
      */
  fromMul(a, b) {
    const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3], a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7], a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11], a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    this[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    this[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    this[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    this[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = b[4];
    b1 = b[5];
    b2 = b[6];
    b3 = b[7];
    this[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    this[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    this[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    this[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = b[8];
    b1 = b[9];
    b2 = b[10];
    b3 = b[11];
    this[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    this[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    this[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    this[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = b[12];
    b1 = b[13];
    b2 = b[14];
    b3 = b[15];
    this[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    this[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    this[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    this[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    return this;
  }
  fromInvert(mat) {
    const a00 = mat[0], a01 = mat[1], a02 = mat[2], a03 = mat[3], a10 = mat[4], a11 = mat[5], a12 = mat[6], a13 = mat[7], a20 = mat[8], a21 = mat[9], a22 = mat[10], a23 = mat[11], a30 = mat[12], a31 = mat[13], a32 = mat[14], a33 = mat[15], b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12, b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det)
      return this;
    det = 1 / det;
    this[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
    this[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
    this[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
    this[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
    this[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
    this[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
    this[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
    this[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
    this[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
    this[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
    this[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
    this[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
    this[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
    this[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
    this[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
    this[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
    return this;
  }
  fromAdjugate(a) {
    const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3], a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7], a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11], a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15], b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12, b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    this[0] = a11 * b11 - a12 * b10 + a13 * b09;
    this[1] = a02 * b10 - a01 * b11 - a03 * b09;
    this[2] = a31 * b05 - a32 * b04 + a33 * b03;
    this[3] = a22 * b04 - a21 * b05 - a23 * b03;
    this[4] = a12 * b08 - a10 * b11 - a13 * b07;
    this[5] = a00 * b11 - a02 * b08 + a03 * b07;
    this[6] = a32 * b02 - a30 * b05 - a33 * b01;
    this[7] = a20 * b05 - a22 * b02 + a23 * b01;
    this[8] = a10 * b10 - a11 * b08 + a13 * b06;
    this[9] = a01 * b08 - a00 * b10 - a03 * b06;
    this[10] = a30 * b04 - a31 * b02 + a33 * b00;
    this[11] = a21 * b02 - a20 * b04 - a23 * b00;
    this[12] = a11 * b07 - a10 * b09 - a12 * b06;
    this[13] = a00 * b09 - a01 * b07 + a02 * b06;
    this[14] = a31 * b01 - a30 * b03 - a32 * b00;
    this[15] = a20 * b03 - a21 * b01 + a22 * b00;
    return this;
  }
  fromFrustum(left, right, bottom, top, near, far) {
    const rl = 1 / (right - left);
    const tb = 1 / (top - bottom);
    const nf = 1 / (near - far);
    this[0] = near * 2 * rl;
    this[1] = 0;
    this[2] = 0;
    this[3] = 0;
    this[4] = 0;
    this[5] = near * 2 * tb;
    this[6] = 0;
    this[7] = 0;
    this[8] = (right + left) * rl;
    this[9] = (top + bottom) * tb;
    this[10] = (far + near) * nf;
    this[11] = -1;
    this[12] = 0;
    this[13] = 0;
    this[14] = far * near * 2 * nf;
    this[15] = 0;
    return this;
  }
  //----------------------------------------------------
  fromQuatTranScale(q, v, s) {
    const x = q[0], y = q[1], z = q[2], w = q[3], x2 = x + x, y2 = y + y, z2 = z + z, xx = x * x2, xy = x * y2, xz = x * z2, yy = y * y2, yz = y * z2, zz = z * z2, wx = w * x2, wy = w * y2, wz = w * z2, sx = s[0], sy = s[1], sz = s[2];
    this[0] = (1 - (yy + zz)) * sx;
    this[1] = (xy + wz) * sx;
    this[2] = (xz - wy) * sx;
    this[3] = 0;
    this[4] = (xy - wz) * sy;
    this[5] = (1 - (xx + zz)) * sy;
    this[6] = (yz + wx) * sy;
    this[7] = 0;
    this[8] = (xz + wy) * sz;
    this[9] = (yz - wx) * sz;
    this[10] = (1 - (xx + yy)) * sz;
    this[11] = 0;
    this[12] = v[0];
    this[13] = v[1];
    this[14] = v[2];
    this[15] = 1;
    return this;
  }
  fromQuatTran(q, v) {
    const x = q[0], y = q[1], z = q[2], w = q[3], x2 = x + x, y2 = y + y, z2 = z + z, xx = x * x2, xy = x * y2, xz = x * z2, yy = y * y2, yz = y * z2, zz = z * z2, wx = w * x2, wy = w * y2, wz = w * z2;
    this[0] = 1 - (yy + zz);
    this[1] = xy + wz;
    this[2] = xz - wy;
    this[3] = 0;
    this[4] = xy - wz;
    this[5] = 1 - (xx + zz);
    this[6] = yz + wx;
    this[7] = 0;
    this[8] = xz + wy;
    this[9] = yz - wx;
    this[10] = 1 - (xx + yy);
    this[11] = 0;
    this[12] = v[0];
    this[13] = v[1];
    this[14] = v[2];
    this[15] = 1;
    return this;
  }
  fromQuat(q) {
    const x = q[0], y = q[1], z = q[2], w = q[3], x2 = x + x, y2 = y + y, z2 = z + z, xx = x * x2, xy = x * y2, xz = x * z2, yy = y * y2, yz = y * z2, zz = z * z2, wx = w * x2, wy = w * y2, wz = w * z2;
    this[0] = 1 - (yy + zz);
    this[1] = xy + wz;
    this[2] = xz - wy;
    this[3] = 0;
    this[4] = xy - wz;
    this[5] = 1 - (xx + zz);
    this[6] = yz + wx;
    this[7] = 0;
    this[8] = xz + wy;
    this[9] = yz - wx;
    this[10] = 1 - (xx + yy);
    this[11] = 0;
    this[12] = 0;
    this[13] = 0;
    this[14] = 0;
    this[15] = 1;
    return this;
  }
  /** Creates a matrix from a quaternion rotation, vector translation and vector scale, rotating and scaling around the given origin */
  fromQuatTranScaleOrigin(q, v, s, o) {
    const x = q[0], y = q[1], z = q[2], w = q[3];
    const x2 = x + x;
    const y2 = y + y;
    const z2 = z + z;
    const xx = x * x2;
    const xy = x * y2;
    const xz = x * z2;
    const yy = y * y2;
    const yz = y * z2;
    const zz = z * z2;
    const wx = w * x2;
    const wy = w * y2;
    const wz = w * z2;
    const sx = s[0];
    const sy = s[1];
    const sz = s[2];
    const ox = o[0];
    const oy = o[1];
    const oz = o[2];
    const out0 = (1 - (yy + zz)) * sx;
    const out1 = (xy + wz) * sx;
    const out2 = (xz - wy) * sx;
    const out4 = (xy - wz) * sy;
    const out5 = (1 - (xx + zz)) * sy;
    const out6 = (yz + wx) * sy;
    const out8 = (xz + wy) * sz;
    const out9 = (yz - wx) * sz;
    const out10 = (1 - (xx + yy)) * sz;
    this[0] = out0;
    this[1] = out1;
    this[2] = out2;
    this[3] = 0;
    this[4] = out4;
    this[5] = out5;
    this[6] = out6;
    this[7] = 0;
    this[8] = out8;
    this[9] = out9;
    this[10] = out10;
    this[11] = 0;
    this[12] = v[0] + ox - (out0 * ox + out4 * oy + out8 * oz);
    this[13] = v[1] + oy - (out1 * ox + out5 * oy + out9 * oz);
    this[14] = v[2] + oz - (out2 * ox + out6 * oy + out10 * oz);
    this[15] = 1;
    return this;
  }
  fromDualQuat(a) {
    const bx = -a[0], by = -a[1], bz = -a[2], bw = a[3], ax = a[4], ay = a[5], az = a[6], aw = a[7];
    const translation = [0, 0, 0];
    let magnitude = bx * bx + by * by + bz * bz + bw * bw;
    if (magnitude > 0) {
      magnitude = 1 / magnitude;
      translation[0] = (ax * bw + aw * bx + ay * bz - az * by) * 2 * magnitude;
      translation[1] = (ay * bw + aw * by + az * bx - ax * bz) * 2 * magnitude;
      translation[2] = (az * bw + aw * bz + ax * by - ay * bx) * 2 * magnitude;
    } else {
      translation[0] = (ax * bw + aw * bx + ay * bz - az * by) * 2;
      translation[1] = (ay * bw + aw * by + az * bx - ax * bz) * 2;
      translation[2] = (az * bw + aw * bz + ax * by - ay * bx) * 2;
    }
    this.fromQuatTran(a, translation);
    return this;
  }
  //----------------------------------------------------
  /** This creates a View Matrix, not a World Matrix. Use fromTarget for a World Matrix */
  fromLook(eye, center, up) {
    let x0, x1, x2, y0, y1, y2, z0, z1, z2, len;
    const eyex = eye[0];
    const eyey = eye[1];
    const eyez = eye[2];
    const upx = up[0];
    const upy = up[1];
    const upz = up[2];
    const centerx = center[0];
    const centery = center[1];
    const centerz = center[2];
    if (Math.abs(eyex - centerx) < 1e-6 && Math.abs(eyey - centery) < 1e-6 && Math.abs(eyez - centerz) < 1e-6) {
      this.identity();
      return this;
    }
    z0 = eyex - centerx;
    z1 = eyey - centery;
    z2 = eyez - centerz;
    len = 1 / Math.sqrt(z0 * z0 + z1 * z1 + z2 * z2);
    z0 *= len;
    z1 *= len;
    z2 *= len;
    x0 = upy * z2 - upz * z1;
    x1 = upz * z0 - upx * z2;
    x2 = upx * z1 - upy * z0;
    len = Math.sqrt(x0 * x0 + x1 * x1 + x2 * x2);
    if (!len) {
      x0 = 0;
      x1 = 0;
      x2 = 0;
    } else {
      len = 1 / len;
      x0 *= len;
      x1 *= len;
      x2 *= len;
    }
    y0 = z1 * x2 - z2 * x1;
    y1 = z2 * x0 - z0 * x2;
    y2 = z0 * x1 - z1 * x0;
    len = Math.sqrt(y0 * y0 + y1 * y1 + y2 * y2);
    if (!len) {
      y0 = 0;
      y1 = 0;
      y2 = 0;
    } else {
      len = 1 / len;
      y0 *= len;
      y1 *= len;
      y2 *= len;
    }
    this[0] = x0;
    this[1] = y0;
    this[2] = z0;
    this[3] = 0;
    this[4] = x1;
    this[5] = y1;
    this[6] = z1;
    this[7] = 0;
    this[8] = x2;
    this[9] = y2;
    this[10] = z2;
    this[11] = 0;
    this[12] = -(x0 * eyex + x1 * eyey + x2 * eyez);
    this[13] = -(y0 * eyex + y1 * eyey + y2 * eyez);
    this[14] = -(z0 * eyex + z1 * eyey + z2 * eyez);
    this[15] = 1;
    return this;
  }
  /** This creates a World Matrix, not a View Matrix. Use fromLook for a View Matrix */
  fromTarget(eye, target, up) {
    const eyex = eye[0], eyey = eye[1], eyez = eye[2], upx = up[0], upy = up[1], upz = up[2];
    let z0 = eyex - target[0], z1 = eyey - target[1], z2 = eyez - target[2], len = z0 * z0 + z1 * z1 + z2 * z2;
    if (len > 0) {
      len = 1 / Math.sqrt(len);
      z0 *= len;
      z1 *= len;
      z2 *= len;
    }
    let x0 = upy * z2 - upz * z1, x1 = upz * z0 - upx * z2, x2 = upx * z1 - upy * z0;
    len = x0 * x0 + x1 * x1 + x2 * x2;
    if (len > 0) {
      len = 1 / Math.sqrt(len);
      x0 *= len;
      x1 *= len;
      x2 *= len;
    }
    this[0] = x0;
    this[1] = x1;
    this[2] = x2;
    this[3] = 0;
    this[4] = z1 * x2 - z2 * x1;
    this[5] = z2 * x0 - z0 * x2;
    this[6] = z0 * x1 - z1 * x0;
    this[7] = 0;
    this[8] = z0;
    this[9] = z1;
    this[10] = z2;
    this[11] = 0;
    this[12] = eyex;
    this[13] = eyey;
    this[14] = eyez;
    this[15] = 1;
    return this;
  }
  //----------------------------------------------------
  fromAxisAngle(axis, rad) {
    let x = axis[0], y = axis[1], z = axis[2], len = Math.hypot(x, y, z);
    if (len < 1e-6)
      return this;
    len = 1 / len;
    x *= len;
    y *= len;
    z *= len;
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    const t = 1 - c;
    this[0] = x * x * t + c;
    this[1] = y * x * t + z * s;
    this[2] = z * x * t - y * s;
    this[3] = 0;
    this[4] = x * y * t - z * s;
    this[5] = y * y * t + c;
    this[6] = z * y * t + x * s;
    this[7] = 0;
    this[8] = x * z * t + y * s;
    this[9] = y * z * t - x * s;
    this[10] = z * z * t + c;
    this[11] = 0;
    this[12] = 0;
    this[13] = 0;
    this[14] = 0;
    this[15] = 1;
    return this;
  }
  fromRotX(rad) {
    const s = Math.sin(rad), c = Math.cos(rad);
    this[0] = 1;
    this[1] = 0;
    this[2] = 0;
    this[3] = 0;
    this[4] = 0;
    this[5] = c;
    this[6] = s;
    this[7] = 0;
    this[8] = 0;
    this[9] = -s;
    this[10] = c;
    this[11] = 0;
    this[12] = 0;
    this[13] = 0;
    this[14] = 0;
    this[15] = 1;
    return this;
  }
  fromRotY(rad) {
    const s = Math.sin(rad), c = Math.cos(rad);
    this[0] = c;
    this[1] = 0;
    this[2] = -s;
    this[3] = 0;
    this[4] = 0;
    this[5] = 1;
    this[6] = 0;
    this[7] = 0;
    this[8] = s;
    this[9] = 0;
    this[10] = c;
    this[11] = 0;
    this[12] = 0;
    this[13] = 0;
    this[14] = 0;
    this[15] = 1;
    return this;
  }
  fromRotZ(rad) {
    const s = Math.sin(rad), c = Math.cos(rad);
    this[0] = c;
    this[1] = s;
    this[2] = 0;
    this[3] = 0;
    this[4] = -s;
    this[5] = c;
    this[6] = 0;
    this[7] = 0;
    this[8] = 0;
    this[9] = 0;
    this[10] = 1;
    this[11] = 0;
    this[12] = 0;
    this[13] = 0;
    this[14] = 0;
    this[15] = 1;
    return this;
  }
  //----------------------------------------------------
  // Calculates a 3x3 normal matrix ( transpose & inverse ) from this 4x4 matrix
  toNormalMat3(out) {
    const a00 = this[0], a01 = this[1], a02 = this[2], a03 = this[3], a10 = this[4], a11 = this[5], a12 = this[6], a13 = this[7], a20 = this[8], a21 = this[9], a22 = this[10], a23 = this[11], a30 = this[12], a31 = this[13], a32 = this[14], a33 = this[15], b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12, b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    out = out || [0, 0, 0, 0, 0, 0, 0, 0, 0];
    if (!det)
      return out;
    det = 1 / det;
    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
    out[1] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
    out[2] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
    out[3] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
    out[4] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
    out[5] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
    out[6] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
    out[7] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
    out[8] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
    return out;
  }
  //----------------------------------------------------
  // FLAT BUFFERS
  /** Used to get data from a flat buffer of matrices */
  fromBuf(ary, idx) {
    this[0] = ary[idx];
    this[1] = ary[idx + 1];
    this[2] = ary[idx + 2];
    this[3] = ary[idx + 3];
    this[4] = ary[idx + 4];
    this[5] = ary[idx + 5];
    this[6] = ary[idx + 6];
    this[7] = ary[idx + 7];
    this[8] = ary[idx + 8];
    this[9] = ary[idx + 9];
    this[10] = ary[idx + 10];
    this[11] = ary[idx + 11];
    this[12] = ary[idx + 12];
    this[13] = ary[idx + 13];
    this[14] = ary[idx + 14];
    this[15] = ary[idx + 15];
    return this;
  }
  /** Put data into a flat buffer of matrices */
  toBuf(ary, idx) {
    ary[idx] = this[0];
    ary[idx + 1] = this[1];
    ary[idx + 2] = this[2];
    ary[idx + 3] = this[3];
    ary[idx + 4] = this[4];
    ary[idx + 5] = this[5];
    ary[idx + 6] = this[6];
    ary[idx + 7] = this[7];
    ary[idx + 8] = this[8];
    ary[idx + 9] = this[9];
    ary[idx + 10] = this[10];
    ary[idx + 11] = this[11];
    ary[idx + 12] = this[12];
    ary[idx + 13] = this[13];
    ary[idx + 14] = this[14];
    ary[idx + 15] = this[15];
    return this;
  }
  // #endregion
  // #region OPERATIONS
  add(b) {
    this[0] = this[0] + b[0];
    this[1] = this[1] + b[1];
    this[2] = this[2] + b[2];
    this[3] = this[3] + b[3];
    this[4] = this[4] + b[4];
    this[5] = this[5] + b[5];
    this[6] = this[6] + b[6];
    this[7] = this[7] + b[7];
    this[8] = this[8] + b[8];
    this[9] = this[9] + b[9];
    this[10] = this[10] + b[10];
    this[11] = this[11] + b[11];
    this[12] = this[12] + b[12];
    this[13] = this[13] + b[13];
    this[14] = this[14] + b[14];
    this[15] = this[15] + b[15];
    return this;
  }
  sub(b) {
    this[0] = this[0] - b[0];
    this[1] = this[1] - b[1];
    this[2] = this[2] - b[2];
    this[3] = this[3] - b[3];
    this[4] = this[4] - b[4];
    this[5] = this[5] - b[5];
    this[6] = this[6] - b[6];
    this[7] = this[7] - b[7];
    this[8] = this[8] - b[8];
    this[9] = this[9] - b[9];
    this[10] = this[10] - b[10];
    this[11] = this[11] - b[11];
    this[12] = this[12] - b[12];
    this[13] = this[13] - b[13];
    this[14] = this[14] - b[14];
    this[15] = this[15] - b[15];
    return this;
  }
  mul(b) {
    const a00 = this[0], a01 = this[1], a02 = this[2], a03 = this[3], a10 = this[4], a11 = this[5], a12 = this[6], a13 = this[7], a20 = this[8], a21 = this[9], a22 = this[10], a23 = this[11], a30 = this[12], a31 = this[13], a32 = this[14], a33 = this[15];
    let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    this[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    this[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    this[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    this[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = b[4];
    b1 = b[5];
    b2 = b[6];
    b3 = b[7];
    this[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    this[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    this[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    this[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = b[8];
    b1 = b[9];
    b2 = b[10];
    b3 = b[11];
    this[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    this[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    this[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    this[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = b[12];
    b1 = b[13];
    b2 = b[14];
    b3 = b[15];
    this[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    this[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    this[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    this[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    return this;
  }
  pmul(b) {
    const a00 = b[0], a01 = b[1], a02 = b[2], a03 = b[3], a10 = b[4], a11 = b[5], a12 = b[6], a13 = b[7], a20 = b[8], a21 = b[9], a22 = b[10], a23 = b[11], a30 = b[12], a31 = b[13], a32 = b[14], a33 = b[15];
    let b0 = this[0], b1 = this[1], b2 = this[2], b3 = this[3];
    this[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    this[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    this[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    this[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = this[4];
    b1 = this[5];
    b2 = this[6];
    b3 = this[7];
    this[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    this[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    this[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    this[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = this[8];
    b1 = this[9];
    b2 = this[10];
    b3 = this[11];
    this[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    this[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    this[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    this[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = this[12];
    b1 = this[13];
    b2 = this[14];
    b3 = this[15];
    this[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    this[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    this[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    this[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    return this;
  }
  invert() {
    const a00 = this[0], a01 = this[1], a02 = this[2], a03 = this[3], a10 = this[4], a11 = this[5], a12 = this[6], a13 = this[7], a20 = this[8], a21 = this[9], a22 = this[10], a23 = this[11], a30 = this[12], a31 = this[13], a32 = this[14], a33 = this[15], b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12, b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det)
      return this;
    det = 1 / det;
    this[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
    this[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
    this[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
    this[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
    this[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
    this[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
    this[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
    this[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
    this[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
    this[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
    this[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
    this[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
    this[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
    this[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
    this[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
    this[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
    return this;
  }
  translate(v, y, z) {
    let xx, yy, zz;
    if (v instanceof Float32Array || v instanceof Array && v.length == 3) {
      xx = v[0];
      yy = v[1];
      zz = v[2];
    } else if (typeof v === "number" && typeof y === "number" && typeof z === "number") {
      xx = v;
      yy = y;
      zz = z;
    } else
      return this;
    this[12] = this[0] * xx + this[4] * yy + this[8] * zz + this[12];
    this[13] = this[1] * xx + this[5] * yy + this[9] * zz + this[13];
    this[14] = this[2] * xx + this[6] * yy + this[10] * zz + this[14];
    this[15] = this[3] * xx + this[7] * yy + this[11] * zz + this[15];
    return this;
  }
  scale(x, y, z) {
    if (y == void 0)
      y = x;
    if (z == void 0)
      z = x;
    this[0] *= x;
    this[1] *= x;
    this[2] *= x;
    this[3] *= x;
    this[4] *= y;
    this[5] *= y;
    this[6] *= y;
    this[7] *= y;
    this[8] *= z;
    this[9] *= z;
    this[10] *= z;
    this[11] *= z;
    return this;
  }
  //----------------------------------------------------
  /** Make the rows into the columns */
  transpose() {
    const a01 = this[1], a02 = this[2], a03 = this[3], a12 = this[6], a13 = this[7], a23 = this[11];
    this[1] = this[4];
    this[2] = this[8];
    this[3] = this[12];
    this[4] = a01;
    this[6] = this[9];
    this[7] = this[13];
    this[8] = a02;
    this[9] = a12;
    this[11] = this[14];
    this[12] = a03;
    this[13] = a13;
    this[14] = a23;
    return this;
  }
  //----------------------------------------------------
  decompose(out_r, out_t, out_s) {
    out_t[0] = this[12];
    out_t[1] = this[13];
    out_t[2] = this[14];
    const m11 = this[0];
    const m12 = this[1];
    const m13 = this[2];
    const m21 = this[4];
    const m22 = this[5];
    const m23 = this[6];
    const m31 = this[8];
    const m32 = this[9];
    const m33 = this[10];
    out_s[0] = Math.hypot(m11, m12, m13);
    out_s[1] = Math.hypot(m21, m22, m23);
    out_s[2] = Math.hypot(m31, m32, m33);
    const is1 = 1 / out_s[0];
    const is2 = 1 / out_s[1];
    const is3 = 1 / out_s[2];
    const sm11 = m11 * is1;
    const sm12 = m12 * is2;
    const sm13 = m13 * is3;
    const sm21 = m21 * is1;
    const sm22 = m22 * is2;
    const sm23 = m23 * is3;
    const sm31 = m31 * is1;
    const sm32 = m32 * is2;
    const sm33 = m33 * is3;
    const trace = sm11 + sm22 + sm33;
    let S = 0;
    if (trace > 0) {
      S = Math.sqrt(trace + 1) * 2;
      out_r[3] = 0.25 * S;
      out_r[0] = (sm23 - sm32) / S;
      out_r[1] = (sm31 - sm13) / S;
      out_r[2] = (sm12 - sm21) / S;
    } else if (sm11 > sm22 && sm11 > sm33) {
      S = Math.sqrt(1 + sm11 - sm22 - sm33) * 2;
      out_r[3] = (sm23 - sm32) / S;
      out_r[0] = 0.25 * S;
      out_r[1] = (sm12 + sm21) / S;
      out_r[2] = (sm31 + sm13) / S;
    } else if (sm22 > sm33) {
      S = Math.sqrt(1 + sm22 - sm11 - sm33) * 2;
      out_r[3] = (sm31 - sm13) / S;
      out_r[0] = (sm12 + sm21) / S;
      out_r[1] = 0.25 * S;
      out_r[2] = (sm23 + sm32) / S;
    } else {
      S = Math.sqrt(1 + sm33 - sm11 - sm22) * 2;
      out_r[3] = (sm12 - sm21) / S;
      out_r[0] = (sm31 + sm13) / S;
      out_r[1] = (sm23 + sm32) / S;
      out_r[2] = 0.25 * S;
    }
    return this;
  }
  //----------------------------------------------------
  rotX(rad) {
    const s = Math.sin(rad), c = Math.cos(rad), a10 = this[4], a11 = this[5], a12 = this[6], a13 = this[7], a20 = this[8], a21 = this[9], a22 = this[10], a23 = this[11];
    this[4] = a10 * c + a20 * s;
    this[5] = a11 * c + a21 * s;
    this[6] = a12 * c + a22 * s;
    this[7] = a13 * c + a23 * s;
    this[8] = a20 * c - a10 * s;
    this[9] = a21 * c - a11 * s;
    this[10] = a22 * c - a12 * s;
    this[11] = a23 * c - a13 * s;
    return this;
  }
  rotY(rad) {
    const s = Math.sin(rad), c = Math.cos(rad), a00 = this[0], a01 = this[1], a02 = this[2], a03 = this[3], a20 = this[8], a21 = this[9], a22 = this[10], a23 = this[11];
    this[0] = a00 * c - a20 * s;
    this[1] = a01 * c - a21 * s;
    this[2] = a02 * c - a22 * s;
    this[3] = a03 * c - a23 * s;
    this[8] = a00 * s + a20 * c;
    this[9] = a01 * s + a21 * c;
    this[10] = a02 * s + a22 * c;
    this[11] = a03 * s + a23 * c;
    return this;
  }
  rotZ(rad) {
    const s = Math.sin(rad), c = Math.cos(rad), a00 = this[0], a01 = this[1], a02 = this[2], a03 = this[3], a10 = this[4], a11 = this[5], a12 = this[6], a13 = this[7];
    this[0] = a00 * c + a10 * s;
    this[1] = a01 * c + a11 * s;
    this[2] = a02 * c + a12 * s;
    this[3] = a03 * c + a13 * s;
    this[4] = a10 * c - a00 * s;
    this[5] = a11 * c - a01 * s;
    this[6] = a12 * c - a02 * s;
    this[7] = a13 * c - a03 * s;
    return this;
  }
  rotAxisAngle(axis, rad) {
    let x = axis[0], y = axis[1], z = axis[2], len = Math.sqrt(x * x + y * y + z * z);
    if (Math.abs(len) < 1e-6)
      return this;
    len = 1 / len;
    x *= len;
    y *= len;
    z *= len;
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    const t = 1 - c;
    const a00 = this[0];
    const a01 = this[1];
    const a02 = this[2];
    const a03 = this[3];
    const a10 = this[4];
    const a11 = this[5];
    const a12 = this[6];
    const a13 = this[7];
    const a20 = this[8];
    const a21 = this[9];
    const a22 = this[10];
    const a23 = this[11];
    const b00 = x * x * t + c;
    const b01 = y * x * t + z * s;
    const b02 = z * x * t - y * s;
    const b10 = x * y * t - z * s;
    const b11 = y * y * t + c;
    const b12 = z * y * t + x * s;
    const b20 = x * z * t + y * s;
    const b21 = y * z * t - x * s;
    const b22 = z * z * t + c;
    this[0] = a00 * b00 + a10 * b01 + a20 * b02;
    this[1] = a01 * b00 + a11 * b01 + a21 * b02;
    this[2] = a02 * b00 + a12 * b01 + a22 * b02;
    this[3] = a03 * b00 + a13 * b01 + a23 * b02;
    this[4] = a00 * b10 + a10 * b11 + a20 * b12;
    this[5] = a01 * b10 + a11 * b11 + a21 * b12;
    this[6] = a02 * b10 + a12 * b11 + a22 * b12;
    this[7] = a03 * b10 + a13 * b11 + a23 * b12;
    this[8] = a00 * b20 + a10 * b21 + a20 * b22;
    this[9] = a01 * b20 + a11 * b21 + a21 * b22;
    this[10] = a02 * b20 + a12 * b21 + a22 * b22;
    this[11] = a03 * b20 + a13 * b21 + a23 * b22;
    return this;
  }
  // #endregion
  // #region TRANSFORMS
  transformVec3(v, out = [0, 0, 0]) {
    const x = v[0], y = v[1], z = v[2];
    out[0] = this[0] * x + this[4] * y + this[8] * z + this[12];
    out[1] = this[1] * x + this[5] * y + this[9] * z + this[13];
    out[2] = this[2] * x + this[6] * y + this[10] * z + this[14];
    return out;
  }
  transformVec4(v, out = [0, 0, 0, 0]) {
    const x = v[0], y = v[1], z = v[2], w = v[3];
    out[0] = this[0] * x + this[4] * y + this[8] * z + this[12] * w;
    out[1] = this[1] * x + this[5] * y + this[9] * z + this[13] * w;
    out[2] = this[2] * x + this[6] * y + this[10] * z + this[14] * w;
    out[3] = this[3] * x + this[7] * y + this[11] * z + this[15] * w;
    return out;
  }
  // #endregion
  // #region STATIC
  static mul(a, b) {
    return new Mat4().fromMul(a, b);
  }
  static invert(a) {
    return new Mat4().fromInvert(a);
  }
  // #endregion 
}

const COMP_LEN = 16;
class MatrixSkin {
  // #region MAIN
  bind;
  world;
  offsetBuffer;
  constructor(bindPose) {
    const bCnt = bindPose.bones.length;
    const mat4Identity = new Mat4();
    const world = new Array(bCnt);
    const bind = new Array(bCnt);
    const preBind = new Mat4();
    this.offsetBuffer = new Float32Array(COMP_LEN * bCnt);
    for (let i = 0; i < bCnt; i++) {
      world[i] = new Mat4();
      bind[i] = new Mat4();
      mat4Identity.toBuf(this.offsetBuffer, i * COMP_LEN);
    }
    let b;
    let l;
    let m = new Mat4();
    preBind.fromQuatTranScale(
      bindPose.offset.rot,
      bindPose.offset.pos,
      bindPose.offset.scl
    );
    for (let i = 0; i < bCnt; i++) {
      b = bindPose.bones[i];
      l = b.local;
      m = world[i];
      m.fromQuatTranScale(l.rot, l.pos, l.scl);
      if (b.pindex !== -1)
        m.pmul(world[b.pindex]);
      else
        m.pmul(preBind);
      bind[i].fromInvert(m);
    }
    this.bind = bind;
    this.world = world;
  }
  // #endregion
  // #region METHODS
  updateFromPose(pose) {
    const offset = new Mat4().fromQuatTranScale(
      pose.offset.rot,
      pose.offset.pos,
      pose.offset.scl
    );
    const bOffset = new Mat4();
    const w = this.world;
    let b;
    let m;
    let i;
    for (i = 0; i < pose.bones.length; i++) {
      b = pose.bones[i];
      m = w[i];
      m.fromQuatTranScale(b.local.rot, b.local.pos, b.local.scl);
      if (b.pindex !== -1)
        m.pmul(w[b.pindex]);
      else
        m.pmul(offset);
      bOffset.fromMul(m, this.bind[i]).toBuf(this.offsetBuffer, i * COMP_LEN);
    }
    return this;
  }
  // #endregion
}

/**
 * Common utilities
 * @module glMatrix
 */
// Configuration Constants
var EPSILON = 0.000001;
var ARRAY_TYPE = typeof Float32Array !== 'undefined' ? Float32Array : Array;
if (!Math.hypot) Math.hypot = function () {
  var y = 0,
      i = arguments.length;

  while (i--) {
    y += arguments[i] * arguments[i];
  }

  return Math.sqrt(y);
};

/**
 * 3x3 Matrix
 * @module mat3
 */

/**
 * Creates a new identity mat3
 *
 * @returns {mat3} a new 3x3 matrix
 */

function create$3() {
  var out = new ARRAY_TYPE(9);

  if (ARRAY_TYPE != Float32Array) {
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[5] = 0;
    out[6] = 0;
    out[7] = 0;
  }

  out[0] = 1;
  out[4] = 1;
  out[8] = 1;
  return out;
}

/**
 * 3 Dimensional Vector
 * @module vec3
 */

/**
 * Creates a new, empty vec3
 *
 * @returns {vec3} a new 3D vector
 */

function create$2() {
  var out = new ARRAY_TYPE(3);

  if (ARRAY_TYPE != Float32Array) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
  }

  return out;
}
/**
 * Calculates the length of a vec3
 *
 * @param {ReadonlyVec3} a vector to calculate length of
 * @returns {Number} length of a
 */

function length(a) {
  var x = a[0];
  var y = a[1];
  var z = a[2];
  return Math.hypot(x, y, z);
}
/**
 * Creates a new vec3 initialized with the given values
 *
 * @param {Number} x X component
 * @param {Number} y Y component
 * @param {Number} z Z component
 * @returns {vec3} a new 3D vector
 */

function fromValues(x, y, z) {
  var out = new ARRAY_TYPE(3);
  out[0] = x;
  out[1] = y;
  out[2] = z;
  return out;
}
/**
 * Copy the values from one vec3 to another
 *
 * @param {vec3} out the receiving vector
 * @param {ReadonlyVec3} a the source vector
 * @returns {vec3} out
 */

function copy(out, a) {
  out[0] = a[0];
  out[1] = a[1];
  out[2] = a[2];
  return out;
}
/**
 * Normalize a vec3
 *
 * @param {vec3} out the receiving vector
 * @param {ReadonlyVec3} a vector to normalize
 * @returns {vec3} out
 */

function normalize$2(out, a) {
  var x = a[0];
  var y = a[1];
  var z = a[2];
  var len = x * x + y * y + z * z;

  if (len > 0) {
    //TODO: evaluate use of glm_invsqrt here?
    len = 1 / Math.sqrt(len);
  }

  out[0] = a[0] * len;
  out[1] = a[1] * len;
  out[2] = a[2] * len;
  return out;
}
/**
 * Calculates the dot product of two vec3's
 *
 * @param {ReadonlyVec3} a the first operand
 * @param {ReadonlyVec3} b the second operand
 * @returns {Number} dot product of a and b
 */

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
/**
 * Computes the cross product of two vec3's
 *
 * @param {vec3} out the receiving vector
 * @param {ReadonlyVec3} a the first operand
 * @param {ReadonlyVec3} b the second operand
 * @returns {vec3} out
 */

function cross(out, a, b) {
  var ax = a[0],
      ay = a[1],
      az = a[2];
  var bx = b[0],
      by = b[1],
      bz = b[2];
  out[0] = ay * bz - az * by;
  out[1] = az * bx - ax * bz;
  out[2] = ax * by - ay * bx;
  return out;
}
/**
 * Alias for {@link vec3.length}
 * @function
 */

var len = length;
/**
 * Perform some operation over an array of vec3s.
 *
 * @param {Array} a the array of vectors to iterate over
 * @param {Number} stride Number of elements between the start of each vec3. If 0 assumes tightly packed
 * @param {Number} offset Number of elements to skip at the beginning of the array
 * @param {Number} count Number of vec3s to iterate over. If 0 iterates over entire array
 * @param {Function} fn Function to call for each vector in the array
 * @param {Object} [arg] additional argument to pass to fn
 * @returns {Array} a
 * @function
 */

(function () {
  var vec = create$2();
  return function (a, stride, offset, count, fn, arg) {
    var i, l;

    if (!stride) {
      stride = 3;
    }

    if (!offset) {
      offset = 0;
    }

    if (count) {
      l = Math.min(count * stride + offset, a.length);
    } else {
      l = a.length;
    }

    for (i = offset; i < l; i += stride) {
      vec[0] = a[i];
      vec[1] = a[i + 1];
      vec[2] = a[i + 2];
      fn(vec, vec, arg);
      a[i] = vec[0];
      a[i + 1] = vec[1];
      a[i + 2] = vec[2];
    }

    return a;
  };
})();

/**
 * 4 Dimensional Vector
 * @module vec4
 */

/**
 * Creates a new, empty vec4
 *
 * @returns {vec4} a new 4D vector
 */

function create$1() {
  var out = new ARRAY_TYPE(4);

  if (ARRAY_TYPE != Float32Array) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
  }

  return out;
}
/**
 * Calculates the squared length of a vec4
 *
 * @param {ReadonlyVec4} a vector to calculate squared length of
 * @returns {Number} squared length of a
 */

function squaredLength$2(a) {
  var x = a[0];
  var y = a[1];
  var z = a[2];
  var w = a[3];
  return x * x + y * y + z * z + w * w;
}
/**
 * Normalize a vec4
 *
 * @param {vec4} out the receiving vector
 * @param {ReadonlyVec4} a vector to normalize
 * @returns {vec4} out
 */

function normalize$1(out, a) {
  var x = a[0];
  var y = a[1];
  var z = a[2];
  var w = a[3];
  var len = x * x + y * y + z * z + w * w;

  if (len > 0) {
    len = 1 / Math.sqrt(len);
  }

  out[0] = x * len;
  out[1] = y * len;
  out[2] = z * len;
  out[3] = w * len;
  return out;
}
/**
 * Perform some operation over an array of vec4s.
 *
 * @param {Array} a the array of vectors to iterate over
 * @param {Number} stride Number of elements between the start of each vec4. If 0 assumes tightly packed
 * @param {Number} offset Number of elements to skip at the beginning of the array
 * @param {Number} count Number of vec4s to iterate over. If 0 iterates over entire array
 * @param {Function} fn Function to call for each vector in the array
 * @param {Object} [arg] additional argument to pass to fn
 * @returns {Array} a
 * @function
 */

(function () {
  var vec = create$1();
  return function (a, stride, offset, count, fn, arg) {
    var i, l;

    if (!stride) {
      stride = 4;
    }

    if (!offset) {
      offset = 0;
    }

    if (count) {
      l = Math.min(count * stride + offset, a.length);
    } else {
      l = a.length;
    }

    for (i = offset; i < l; i += stride) {
      vec[0] = a[i];
      vec[1] = a[i + 1];
      vec[2] = a[i + 2];
      vec[3] = a[i + 3];
      fn(vec, vec, arg);
      a[i] = vec[0];
      a[i + 1] = vec[1];
      a[i + 2] = vec[2];
      a[i + 3] = vec[3];
    }

    return a;
  };
})();

/**
 * Quaternion
 * @module quat
 */

/**
 * Creates a new identity quat
 *
 * @returns {quat} a new quaternion
 */

function create() {
  var out = new ARRAY_TYPE(4);

  if (ARRAY_TYPE != Float32Array) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
  }

  out[3] = 1;
  return out;
}
/**
 * Sets a quat from the given angle and rotation axis,
 * then returns it.
 *
 * @param {quat} out the receiving quaternion
 * @param {ReadonlyVec3} axis the axis around which to rotate
 * @param {Number} rad the angle in radians
 * @returns {quat} out
 **/

function setAxisAngle(out, axis, rad) {
  rad = rad * 0.5;
  var s = Math.sin(rad);
  out[0] = s * axis[0];
  out[1] = s * axis[1];
  out[2] = s * axis[2];
  out[3] = Math.cos(rad);
  return out;
}
/**
 * Performs a spherical linear interpolation between two quat
 *
 * @param {quat} out the receiving quaternion
 * @param {ReadonlyQuat} a the first operand
 * @param {ReadonlyQuat} b the second operand
 * @param {Number} t interpolation amount, in the range [0-1], between the two inputs
 * @returns {quat} out
 */

function slerp(out, a, b, t) {
  // benchmarks:
  //    http://jsperf.com/quaternion-slerp-implementations
  var ax = a[0],
      ay = a[1],
      az = a[2],
      aw = a[3];
  var bx = b[0],
      by = b[1],
      bz = b[2],
      bw = b[3];
  var omega, cosom, sinom, scale0, scale1; // calc cosine

  cosom = ax * bx + ay * by + az * bz + aw * bw; // adjust signs (if necessary)

  if (cosom < 0.0) {
    cosom = -cosom;
    bx = -bx;
    by = -by;
    bz = -bz;
    bw = -bw;
  } // calculate coefficients


  if (1.0 - cosom > EPSILON) {
    // standard case (slerp)
    omega = Math.acos(cosom);
    sinom = Math.sin(omega);
    scale0 = Math.sin((1.0 - t) * omega) / sinom;
    scale1 = Math.sin(t * omega) / sinom;
  } else {
    // "from" and "to" quaternions are very close
    //  ... so we can do a linear interpolation
    scale0 = 1.0 - t;
    scale1 = t;
  } // calculate final values


  out[0] = scale0 * ax + scale1 * bx;
  out[1] = scale0 * ay + scale1 * by;
  out[2] = scale0 * az + scale1 * bz;
  out[3] = scale0 * aw + scale1 * bw;
  return out;
}
/**
 * Creates a quaternion from the given 3x3 rotation matrix.
 *
 * NOTE: The resultant quaternion is not normalized, so you should be sure
 * to renormalize the quaternion yourself where necessary.
 *
 * @param {quat} out the receiving quaternion
 * @param {ReadonlyMat3} m rotation matrix
 * @returns {quat} out
 * @function
 */

function fromMat3(out, m) {
  // Algorithm in Ken Shoemake's article in 1987 SIGGRAPH course notes
  // article "Quaternion Calculus and Fast Animation".
  var fTrace = m[0] + m[4] + m[8];
  var fRoot;

  if (fTrace > 0.0) {
    // |w| > 1/2, may as well choose w > 1/2
    fRoot = Math.sqrt(fTrace + 1.0); // 2w

    out[3] = 0.5 * fRoot;
    fRoot = 0.5 / fRoot; // 1/(4w)

    out[0] = (m[5] - m[7]) * fRoot;
    out[1] = (m[6] - m[2]) * fRoot;
    out[2] = (m[1] - m[3]) * fRoot;
  } else {
    // |w| <= 1/2
    var i = 0;
    if (m[4] > m[0]) i = 1;
    if (m[8] > m[i * 3 + i]) i = 2;
    var j = (i + 1) % 3;
    var k = (i + 2) % 3;
    fRoot = Math.sqrt(m[i * 3 + i] - m[j * 3 + j] - m[k * 3 + k] + 1.0);
    out[i] = 0.5 * fRoot;
    fRoot = 0.5 / fRoot;
    out[3] = (m[j * 3 + k] - m[k * 3 + j]) * fRoot;
    out[j] = (m[j * 3 + i] + m[i * 3 + j]) * fRoot;
    out[k] = (m[k * 3 + i] + m[i * 3 + k]) * fRoot;
  }

  return out;
}
/**
 * Calculates the squared length of a quat
 *
 * @param {ReadonlyQuat} a vector to calculate squared length of
 * @returns {Number} squared length of a
 * @function
 */

var squaredLength$1 = squaredLength$2;
/**
 * Normalize a quat
 *
 * @param {quat} out the receiving quaternion
 * @param {ReadonlyQuat} a quaternion to normalize
 * @returns {quat} out
 * @function
 */

var normalize = normalize$1;
/**
 * Sets a quaternion to represent the shortest rotation from one
 * vector to another.
 *
 * Both vectors are assumed to be unit length.
 *
 * @param {quat} out the receiving quaternion.
 * @param {ReadonlyVec3} a the initial vector
 * @param {ReadonlyVec3} b the destination vector
 * @returns {quat} out
 */

(function () {
  var tmpvec3 = create$2();
  var xUnitVec3 = fromValues(1, 0, 0);
  var yUnitVec3 = fromValues(0, 1, 0);
  return function (out, a, b) {
    var dot$1 = dot(a, b);

    if (dot$1 < -0.999999) {
      cross(tmpvec3, xUnitVec3, a);
      if (len(tmpvec3) < 0.000001) cross(tmpvec3, yUnitVec3, a);
      normalize$2(tmpvec3, tmpvec3);
      setAxisAngle(out, tmpvec3, Math.PI);
      return out;
    } else if (dot$1 > 0.999999) {
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
      out[3] = 1;
      return out;
    } else {
      cross(tmpvec3, a, b);
      out[0] = tmpvec3[0];
      out[1] = tmpvec3[1];
      out[2] = tmpvec3[2];
      out[3] = 1 + dot$1;
      return normalize(out, out);
    }
  };
})();
/**
 * Performs a spherical linear interpolation with two control points
 *
 * @param {quat} out the receiving quaternion
 * @param {ReadonlyQuat} a the first operand
 * @param {ReadonlyQuat} b the second operand
 * @param {ReadonlyQuat} c the third operand
 * @param {ReadonlyQuat} d the fourth operand
 * @param {Number} t interpolation amount, in the range [0-1], between the two inputs
 * @returns {quat} out
 */

(function () {
  var temp1 = create();
  var temp2 = create();
  return function (out, a, b, c, d, t) {
    slerp(temp1, a, d, t);
    slerp(temp2, b, c, t);
    slerp(out, temp1, temp2, 2 * t * (1 - t));
    return out;
  };
})();
/**
 * Sets the specified quaternion with values corresponding to the given
 * axes. Each axis is a vec3 and is expected to be unit length and
 * perpendicular to all other specified axes.
 *
 * @param {ReadonlyVec3} view  the vector representing the viewing direction
 * @param {ReadonlyVec3} right the vector representing the local "right" direction
 * @param {ReadonlyVec3} up    the vector representing the local "up" direction
 * @returns {quat} out
 */

(function () {
  var matr = create$3();
  return function (out, view, right, up) {
    matr[0] = right[0];
    matr[3] = right[1];
    matr[6] = right[2];
    matr[1] = up[0];
    matr[4] = up[1];
    matr[7] = up[2];
    matr[2] = -view[0];
    matr[5] = -view[1];
    matr[8] = -view[2];
    return normalize(out, fromMat3(out, matr));
  };
})();

/**
 * Creates a dual quat from a quaternion and a translation
 *
 * @param {ReadonlyQuat2} dual quaternion receiving operation result
 * @param {ReadonlyQuat} q a normalized quaternion
 * @param {ReadonlyVec3} t tranlation vector
 * @returns {quat2} dual quaternion receiving operation result
 * @function
 */

function fromRotationTranslation(out, q, t) {
  var ax = t[0] * 0.5,
      ay = t[1] * 0.5,
      az = t[2] * 0.5,
      bx = q[0],
      by = q[1],
      bz = q[2],
      bw = q[3];
  out[0] = bx;
  out[1] = by;
  out[2] = bz;
  out[3] = bw;
  out[4] = ax * bw + ay * bz - az * by;
  out[5] = ay * bw + az * bx - ax * bz;
  out[6] = az * bw + ax * by - ay * bx;
  out[7] = -ax * bx - ay * by - az * bz;
  return out;
}
/**
 * Multiplies two dual quat's
 *
 * @param {quat2} out the receiving dual quaternion
 * @param {ReadonlyQuat2} a the first operand
 * @param {ReadonlyQuat2} b the second operand
 * @returns {quat2} out
 */

function multiply(out, a, b) {
  var ax0 = a[0],
      ay0 = a[1],
      az0 = a[2],
      aw0 = a[3],
      bx1 = b[4],
      by1 = b[5],
      bz1 = b[6],
      bw1 = b[7],
      ax1 = a[4],
      ay1 = a[5],
      az1 = a[6],
      aw1 = a[7],
      bx0 = b[0],
      by0 = b[1],
      bz0 = b[2],
      bw0 = b[3];
  out[0] = ax0 * bw0 + aw0 * bx0 + ay0 * bz0 - az0 * by0;
  out[1] = ay0 * bw0 + aw0 * by0 + az0 * bx0 - ax0 * bz0;
  out[2] = az0 * bw0 + aw0 * bz0 + ax0 * by0 - ay0 * bx0;
  out[3] = aw0 * bw0 - ax0 * bx0 - ay0 * by0 - az0 * bz0;
  out[4] = ax0 * bw1 + aw0 * bx1 + ay0 * bz1 - az0 * by1 + ax1 * bw0 + aw1 * bx0 + ay1 * bz0 - az1 * by0;
  out[5] = ay0 * bw1 + aw0 * by1 + az0 * bx1 - ax0 * bz1 + ay1 * bw0 + aw1 * by0 + az1 * bx0 - ax1 * bz0;
  out[6] = az0 * bw1 + aw0 * bz1 + ax0 * by1 - ay0 * bx1 + az1 * bw0 + aw1 * bz0 + ax1 * by0 - ay1 * bx0;
  out[7] = aw0 * bw1 - ax0 * bx1 - ay0 * by1 - az0 * bz1 + aw1 * bw0 - ax1 * bx0 - ay1 * by0 - az1 * bz0;
  return out;
}
/**
 * Alias for {@link quat2.multiply}
 * @function
 */

var mul = multiply;
/**
 * Calculates the inverse of a dual quat. If they are normalized, conjugate is cheaper
 *
 * @param {quat2} out the receiving dual quaternion
 * @param {ReadonlyQuat2} a dual quat to calculate inverse of
 * @returns {quat2} out
 */

function invert(out, a) {
  var sqlen = squaredLength(a);
  out[0] = -a[0] / sqlen;
  out[1] = -a[1] / sqlen;
  out[2] = -a[2] / sqlen;
  out[3] = a[3] / sqlen;
  out[4] = -a[4] / sqlen;
  out[5] = -a[5] / sqlen;
  out[6] = -a[6] / sqlen;
  out[7] = a[7] / sqlen;
  return out;
}
/**
 * Calculates the squared length of a dual quat
 *
 * @param {ReadonlyQuat2} a dual quat to calculate squared length of
 * @returns {Number} squared length of a
 * @function
 */

var squaredLength = squaredLength$1;

class Vec3Ex {
  // #region LOADING / CONVERSION
  /** Used to get data from a flat buffer */
  static fromBuf(out, ary, idx) {
    out[0] = ary[idx];
    out[1] = ary[idx + 1];
    out[2] = ary[idx + 2];
    return out;
  }
  /** Put data into a flat buffer */
  static toBuf(v, ary, idx) {
    ary[idx] = v[0];
    ary[idx + 1] = v[1];
    ary[idx + 2] = v[2];
    return this;
  }
  // #endregion
  static lookAxes(dir, up = [0, 1, 0], xAxis = [1, 0, 0], yAxis = [0, 1, 0], zAxis = [0, 0, 1]) {
    copy(zAxis, dir);
    cross(xAxis, up, zAxis);
    cross(yAxis, zAxis, xAxis);
    normalize$2(xAxis, xAxis);
    normalize$2(yAxis, yAxis);
    normalize$2(zAxis, zAxis);
  }
  static project(out, from, to) {
    const denom = dot(to, to);
    if (denom < 1e-6) {
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
      return out;
    }
    const scl = dot(from, to) / denom;
    out[0] = to[0] * scl;
    out[1] = to[1] * scl;
    out[2] = to[2] * scl;
    return out;
  }
}

let DQTSkin$1 = class DQTSkin {
  // #region MAIN
  bind;
  world;
  // Split into 3 Buffers because THREEJS does handle mat4x3 correctly
  // Since using in Shader Uniforms, can skip the 16 byte alignment for scale & store data as Vec3 instead of Vec4.
  // TODO : This may change in the future into a single mat4x3 buffer.
  offsetQBuffer;
  // DualQuat : Quaternion
  offsetPBuffer;
  // DualQuat : Translation
  offsetSBuffer;
  // Scale
  constructor(arm) {
    const bCnt = arm.boneCount;
    const world = new Array(bCnt);
    const bind = new Array(bCnt);
    this.offsetQBuffer = new Float32Array(4 * bCnt);
    this.offsetPBuffer = new Float32Array(4 * bCnt);
    this.offsetSBuffer = new Float32Array(3 * bCnt);
    for (let i = 0; i < bCnt; i++) {
      world[i] = new Transform();
      bind[i] = new Transform();
      Vec3Ex.toBuf([1, 1, 1], this.offsetSBuffer, i * 3);
    }
    const pose = arm.bindPose;
    let b;
    for (let i = 0; i < bCnt; i++) {
      b = pose.bones[i];
      if (b.pindex !== -1)
        world[i].fromMul(world[b.pindex], b.local);
      else
        world[i].copy(b.local);
      bind[i].fromInvert(world[i]);
    }
    this.bind = bind;
    this.world = world;
  }
  // #endregion
  // #region METHODS
  updateFromPose(pose) {
    const bOffset = new Transform();
    const w = this.world;
    const dq = [0, 0, 0, 1, 0, 0, 0, 0];
    let b;
    let ii = 0;
    let si = 0;
    for (let i = 0; i < pose.bones.length; i++) {
      b = pose.bones[i];
      if (b.pindex !== -1)
        w[i].fromMul(w[b.pindex], b.local);
      else
        w[i].fromMul(pose.offset, b.local);
      bOffset.fromMul(w[i], this.bind[i]);
      fromRotationTranslation(dq, bOffset.rot, bOffset.pos);
      ii = i * 4;
      si = i * 3;
      this.offsetQBuffer[ii + 0] = dq[0];
      this.offsetQBuffer[ii + 1] = dq[1];
      this.offsetQBuffer[ii + 2] = dq[2];
      this.offsetQBuffer[ii + 3] = dq[3];
      this.offsetPBuffer[ii + 0] = dq[4];
      this.offsetPBuffer[ii + 1] = dq[5];
      this.offsetPBuffer[ii + 2] = dq[6];
      this.offsetPBuffer[ii + 3] = dq[7];
      this.offsetSBuffer[si + 0] = bOffset.scl[0];
      this.offsetSBuffer[si + 1] = bOffset.scl[1];
      this.offsetSBuffer[si + 2] = bOffset.scl[2];
    }
    return this;
  }
  // #endregion
};

class TranMatrixSkin {
  // #region MAIN
  bind;
  // Bind pose 
  world;
  // World space computation
  offsetBuffer;
  // Final Output for shaders to use
  constructor(bindPose) {
    const bCnt = bindPose.bones.length;
    const mat4Identity = new Mat4();
    const world = new Array(bCnt);
    const bind = new Array(bCnt);
    this.offsetBuffer = new Float32Array(16 * bCnt);
    for (let i = 0; i < bCnt; i++) {
      world[i] = new Transform();
      bind[i] = new Transform();
      mat4Identity.toBuf(this.offsetBuffer, i * 16);
    }
    let b;
    for (let i = 0; i < bCnt; i++) {
      b = bindPose.bones[i];
      if (b.pindex !== -1)
        world[i].fromMul(world[b.pindex], b.local);
      else
        world[i].copy(b.local);
      bind[i].fromInvert(world[i]);
    }
    this.bind = bind;
    this.world = world;
  }
  // #endregion
  // #region METHODS
  updateFromPose(pose) {
    const bOffset = new Transform();
    const w = this.world;
    const m = new Mat4();
    let b;
    for (let i = 0; i < pose.bones.length; i++) {
      b = pose.bones[i];
      if (b.pindex !== -1)
        w[i].fromMul(w[b.pindex], b.local);
      else
        w[i].fromMul(pose.offset, b.local);
      bOffset.fromMul(w[i], this.bind[i]);
      m.fromQuatTranScale(bOffset.rot, bOffset.pos, bOffset.scl).toBuf(this.offsetBuffer, i * 16);
    }
    return this;
  }
  // #endregion
}

class DQTSkin {
  // #region MAIN
  bind;
  world;
  // Split into 2 Buffers because THREEJS does handle mat4x2 correctly
  offsetQBuffer;
  // DualQuat : Quaternion
  offsetPBuffer;
  // DualQuat : Translation
  constructor(arm) {
    const bCnt = arm.boneCount;
    const world = new Array(bCnt);
    const bind = new Array(bCnt);
    this.offsetQBuffer = new Float32Array(4 * bCnt);
    this.offsetPBuffer = new Float32Array(4 * bCnt);
    for (let i = 0; i < bCnt; i++) {
      world[i] = [0, 0, 0, 1, 0, 0, 0, 0];
      bind[i] = [0, 0, 0, 1, 0, 0, 0, 0];
    }
    const pose = arm.bindPose;
    let b;
    for (let i = 0; i < bCnt; i++) {
      b = pose.bones[i];
      fromRotationTranslation(world[i], b.local.rot, b.local.pos);
      if (b.pindex !== -1)
        mul(world[i], world[b.pindex], world[i]);
      invert(bind[i], world[i]);
    }
    this.bind = bind;
    this.world = world;
  }
  // #endregion
  // #region METHODS
  updateFromPose(pose) {
    const offset = fromRotationTranslation([0, 0, 0, 1, 0, 0, 0, 0], pose.offset.rot, pose.offset.pos);
    const dq = [0, 0, 0, 1, 0, 0, 0, 0];
    const w = this.world;
    let b;
    let ii = 0;
    for (let i = 0; i < pose.bones.length; i++) {
      b = pose.bones[i];
      fromRotationTranslation(dq, b.local.rot, b.local.pos);
      if (b.pindex !== -1)
        mul(w[i], w[b.pindex], dq);
      else
        mul(w[i], offset, dq);
      mul(dq, w[i], this.bind[i]);
      ii = i * 4;
      this.offsetQBuffer[ii + 0] = dq[0];
      this.offsetQBuffer[ii + 1] = dq[1];
      this.offsetQBuffer[ii + 2] = dq[2];
      this.offsetQBuffer[ii + 3] = dq[3];
      this.offsetPBuffer[ii + 0] = dq[4];
      this.offsetPBuffer[ii + 1] = dq[5];
      this.offsetPBuffer[ii + 2] = dq[6];
      this.offsetPBuffer[ii + 3] = dq[7];
    }
    return this;
  }
  // #endregion
}

class SQTSkin {
  // #region MAIN
  bind;
  world;
  // Split into 3 Buffers because THREEJS does handle mat4x3 correctly
  // Since using in Shader Uniforms, can skip the 16 byte alignment for scale + pos, store data as Vec3 instead of Vec4.
  offsetQBuffer;
  // Quaternion
  offsetPBuffer;
  // Translation
  offsetSBuffer;
  // Scale
  constructor(bindPose) {
    const bCnt = bindPose.bones.length;
    const world = new Array(bCnt);
    const bind = new Array(bCnt);
    this.offsetQBuffer = new Float32Array(4 * bCnt);
    this.offsetPBuffer = new Float32Array(3 * bCnt);
    this.offsetSBuffer = new Float32Array(3 * bCnt);
    const vScl = new Vec3(1, 1, 1);
    const vPos = new Vec3();
    const vRot = new Quat();
    for (let i = 0; i < bCnt; i++) {
      world[i] = new Transform();
      bind[i] = new Transform();
      vRot.toBuf(this.offsetQBuffer, i * 4);
      vPos.toBuf(this.offsetPBuffer, i * 3);
      vScl.toBuf(this.offsetSBuffer, i * 3);
    }
    let b;
    for (let i = 0; i < bCnt; i++) {
      b = bindPose.bones[i];
      if (b.pindex !== -1)
        world[i].fromMul(world[b.pindex], b.local);
      else
        world[i].copy(b.local);
      bind[i].fromInvert(world[i]);
    }
    this.bind = bind;
    this.world = world;
  }
  // #endregion
  // #region METHODS
  updateFromPose(pose) {
    const bOffset = new Transform();
    const w = this.world;
    let b;
    let ii = 0;
    let si = 0;
    for (let i = 0; i < pose.bones.length; i++) {
      b = pose.bones[i];
      if (b.pindex !== -1)
        w[i].fromMul(w[b.pindex], b.local);
      else
        w[i].fromMul(pose.offset, b.local);
      bOffset.fromMul(w[i], this.bind[i]);
      ii = i * 4;
      si = i * 3;
      this.offsetQBuffer[ii + 0] = bOffset.rot[0];
      this.offsetQBuffer[ii + 1] = bOffset.rot[1];
      this.offsetQBuffer[ii + 2] = bOffset.rot[2];
      this.offsetQBuffer[ii + 3] = bOffset.rot[3];
      this.offsetPBuffer[si + 0] = bOffset.pos[0];
      this.offsetPBuffer[si + 1] = bOffset.pos[1];
      this.offsetPBuffer[si + 2] = bOffset.pos[2];
      this.offsetSBuffer[si + 0] = bOffset.scl[0];
      this.offsetSBuffer[si + 1] = bOffset.scl[1];
      this.offsetSBuffer[si + 2] = bOffset.scl[2];
    }
    return this;
  }
  // #endregion
}

class BoneAxes {
  // #region STATIC
  // SWING-TWIST-ORTHO - What is Forward - Up - Left
  static UFR = 0;
  // Aim, Chest, BLENDER LIKE
  static RBD = 1;
  // Left Arm
  static LBU = 2;
  // Right Arm
  static DFR = 3;
  // Legs
  static FUR = 4;
  // Standard WorldSpace Dir
  // #endregion
  // #region MAIN
  swing = new Vec3(Vec3.UP);
  // Y
  twist = new Vec3(Vec3.FORWARD);
  // Z
  ortho = new Vec3(Vec3.RIGHT);
  // X
  constructor(axes) {
    if (axes)
      this.copy(axes);
  }
  // #endregion
  // #region METHODS
  /** Get new BoneAxes with quaternion applied */
  getFromQuat(q, rtn = new BoneAxes()) {
    return rtn.copy(this).applyQuat(q);
  }
  copy(axes) {
    this.swing.copy(axes.swing);
    this.twist.copy(axes.twist);
    this.ortho.copy(axes.ortho);
    return this;
  }
  /** Apply rotation on current axes directions */
  applyQuat(q) {
    this.swing.fromQuat(q, this.swing).norm();
    this.twist.fromQuat(q, this.twist).norm();
    this.ortho.fromQuat(q, this.ortho).norm();
    return this;
  }
  /** Set a quaternion direction set. For swing-twist ik rotations
   * please make sure the quaternion passed in are inverted first
   * when your looking to support qi axes.
   */
  setQuatDirections(q, ba = BoneAxes.UFR) {
    switch (ba) {
      case BoneAxes.UFR:
        this.swing.fromQuat(q, Vec3.UP);
        this.twist.fromQuat(q, Vec3.FORWARD);
        this.ortho.fromQuat(q, Vec3.RIGHT);
        break;
      case BoneAxes.RBD:
        this.swing.fromQuat(q, Vec3.RIGHT);
        this.twist.fromQuat(q, Vec3.BACK);
        this.ortho.fromQuat(q, Vec3.DOWN);
        break;
      case BoneAxes.LBU:
        this.swing.fromQuat(q, Vec3.LEFT);
        this.twist.fromQuat(q, Vec3.BACK);
        this.ortho.fromQuat(q, Vec3.UP);
        break;
      case BoneAxes.DFR:
        this.swing.fromQuat(q, Vec3.DOWN);
        this.twist.fromQuat(q, Vec3.FORWARD);
        this.ortho.fromQuat(q, Vec3.RIGHT);
        break;
      case BoneAxes.FUR:
        this.swing.fromQuat(q, Vec3.FORWARD);
        this.twist.fromQuat(q, Vec3.UP);
        this.ortho.fromQuat(q, Vec3.RIGHT);
        break;
    }
    this.swing.norm();
    this.twist.norm();
    this.ortho.norm();
    return this;
  }
  // #endregion
}

class IKTarget {
  // #region MAIN
  hasChanged = false;
  tMode = 0;
  // Initial Target : Position or Direction
  pMode = 0;
  // Initial Pole   : Position or Direction
  deltaMove = new Vec3();
  // How much to move a bone
  endPos = new Vec3();
  // Target Position
  startPos = new Vec3();
  // Origin Position
  polePos = new Vec3();
  // Position of Pole
  dist = 0;
  // Distance from Origin & Target Position
  swing = new Vec3();
  // To Target Direction or end-start position
  twist = new Vec3();
  // To Pole Direction or Orth direction of swing
  lenScale = -1;
  // How to scale the swing direction when computing IK Target Position
  altSwing;
  // Second set of SwingTwist Directions
  altTwist;
  // ... used just for SwingTwistEnds Solver
  pworld = new Transform();
  // Parent Bone WS Transform
  rworld = new Transform();
  // Root Bone WS Transform
  // #endregion
  // #region SETTERS
  setPositions(t, p) {
    this.hasChanged = true;
    this.tMode = 0;
    this.pMode = 0;
    this.endPos.copy(t);
    if (p)
      this.polePos.copy(p);
    return this;
  }
  setDirections(s, t, scl) {
    this.hasChanged = true;
    this.swing.copy(s);
    this.tMode = 1;
    if (t) {
      this.twist.copy(t);
      this.pMode = 1;
    } else
      this.pMode = 0;
    if (scl)
      this.lenScale = scl;
    return this;
  }
  setAltDirections(s, t) {
    this.hasChanged = true;
    if (!this.altSwing) {
      this.altSwing = new Vec3();
      this.altTwist = new Vec3();
    }
    this.altSwing.copy(s);
    this.altTwist.copy(t);
    return this;
  }
  setPoleDir(p) {
    this.hasChanged = true;
    this.pMode = 1;
    this.twist.copy(p);
    return this;
  }
  setDeltaMove(p, scl = 1) {
    this.deltaMove.copy(p).scale(scl);
    this.hasChanged = true;
    return this;
  }
  // #endregion
  // #region METHODS
  resolveTarget(chain, pose) {
    pose.getWorldTransform(chain.links[0].pindex, this.pworld);
    this.rworld.fromMul(this.pworld, chain.links[0].bind);
    switch (this.tMode) {
      case 0:
        this.startPos.copy(this.rworld.pos);
        this.swing.fromSub(this.endPos, this.startPos);
        this.dist = this.swing.len;
        this.swing.norm();
        break;
      case 1:
        this.dist = this.lenScale >= 0 ? this.lenScale * chain.len : chain.len;
        this.startPos.copy(this.rworld.pos);
        this.endPos.copy(this.swing).scale(this.dist).add(this.rworld.pos);
        break;
    }
    switch (this.pMode) {
      case 0:
        this.twist.fromSub(this.polePos, this.startPos).alignTwist(this.swing, this.twist).norm();
        break;
    }
    this.hasChanged = false;
    return this;
  }
  // #endregion
  // #region HELPERS
  debug(d) {
    d.pnt.add(this.startPos, 16777215, 3, 0);
    d.pnt.add(this.endPos, 16777215, 3, 1);
    d.ln.add(this.startPos, this.endPos, 16777215);
    const p = this.twist.clone().scale(0.5).add(this.startPos);
    d.ln.add(this.startPos, p, 16777215);
    d.pnt.add(p, 16777215, 3, 6);
    return this;
  }
  // #endregion
}

class IKLink {
  index = -1;
  // Bone Index
  pindex = -1;
  // Parent Bone Index
  len = 0;
  // Computed Length of the bone in chain
  axes = new BoneAxes();
  // IK May need alternative axis directions
  bind = new Transform();
  // Bind Link to a specific pose, localspace transform
  constructor(bone, swingTwist = -1) {
    this.index = bone.index;
    this.pindex = bone.pindex;
    this.bind.copy(bone.local);
    if (swingTwist !== -1) {
      this.axes.setQuatDirections(
        new Quat().fromInvert(bone.world.rot),
        swingTwist
      );
    }
  }
}
class IKChain {
  // #region MAIN
  links = [];
  // List of linked bones
  len = 0;
  // Length of the chain
  constructor(bones, swingTwist = -1) {
    if (bones)
      this.setBones(bones, swingTwist);
  }
  // #endregion
  // #region GETTERS // SETTERS
  get lastLink() {
    return this.links[this.links.length - 1];
  }
  get firstLink() {
    return this.links[0];
  }
  setBones(bones, swingTwist = -1) {
    this.links.length = 0;
    this.len = 0;
    for (const b of bones)
      this.links.push(new IKLink(b, swingTwist));
    const li = bones.length - 1;
    for (let i = 1; i <= li; i++) {
      this.links[i - 1].len = Vec3.dist(bones[i].world.pos, bones[i - 1].world.pos);
      this.len += this.links[i - 1].len;
    }
    this.links[li].len = bones[li].len;
    return this;
  }
  /** Reset linked bones with the bind transfrom saved in the chain */
  resetPoseLocal(pose, startIdx = 0, endIdx = -1) {
    if (endIdx < 0)
      endIdx = this.links.length - 1;
    let lnk;
    for (let i = startIdx; i <= endIdx; i++) {
      lnk = this.links[i];
      pose.bones[lnk.index].local.rot.copy(lnk.bind.rot);
    }
    return this;
  }
  debug(debug, pose) {
    const t = new Transform();
    const a = new BoneAxes();
    const v = new Vec3();
    for (const l of this.links) {
      pose.getWorldTransform(l.index, t);
      l.axes.getFromQuat(t.rot, a);
      debug.pnt.add(t.pos, 16777215, 1, 0);
      debug.ln.add(t.pos, v.fromScaleThenAdd(0.1, a.swing, t.pos), 16777215);
      debug.ln.add(t.pos, v.fromScaleThenAdd(0.1, a.twist, t.pos), 16711935);
      debug.ln.add(t.pos, v.fromScaleThenAdd(0.1, a.ortho, t.pos), 7368816);
    }
  }
  /** Simplify getting a pose bone. No index will return root bone */
  // getPoseBone( pose: Pose, idx: number = 0 ): Bone{ return pose.getBone( chain.links[ idx ].index ); }
  // getPoseWorldBind( pose: Pose, idx: number =0, pTran: Transform = new Transform() ){
  //     pose.getWorldTransform( chain.links[ idx ].pindex, pTran );
  //     return pTran.clone().mul( chain.links[ idx ].Bind );
  // }
  // #endregion
}

function lookSolver(tar, chain, pose, Debug) {
  const lnk = chain.links[0];
  const tDir = new Vec3();
  tDir.fromQuat(tar.rworld.rot, lnk.axes.swing);
  const rot = new Quat().fromSwing(tDir, tar.swing).mul(tar.rworld.rot);
  tDir.fromQuat(rot, lnk.axes.twist);
  if (Vec3.dot(tar.twist, tDir) < 0.999999) {
    const twistReset = new Quat().fromSwing(tDir, tar.twist);
    if (Vec3.dot(twistReset, rot) < 0)
      twistReset.negate();
    rot.pmul(twistReset);
  }
  rot.pmulInvert(tar.pworld.rot);
  pose.setLocalRot(lnk.index, rot);
  if (Debug) {
    Debug.ln.add(tar.startPos, new Vec3().fromAdd(tDir, tar.startPos), 65280);
    Debug.ln.add(tar.startPos, new Vec3().fromAdd(tar.twist, tar.startPos), 16777215);
    Debug.ln.add(tar.startPos, tar.endPos, 16777215);
  }
}

function deltaMoveSolver(tar, chain, pose) {
  const pTran = new Transform();
  const cTran = new Transform();
  const ptInv = new Transform();
  const lnk = chain.firstLink;
  pose.getWorldTransform(lnk.pindex, pTran);
  cTran.fromMul(pTran, lnk.bind);
  ptInv.fromInvert(pTran);
  cTran.pos.add(tar.deltaMove);
  ptInv.transformVec3(cTran.pos);
  pose.setLocalPos(lnk.index, cTran.pos);
}

function rootCompose(target, chain, pose) {
  target.resolveTarget(chain, pose);
  lookSolver(target, chain, pose);
  deltaMoveSolver(target, chain, pose);
}

function limbCompose$1(target, chain, pose) {
  target.resolveTarget(chain, pose);
  lookSolver(target, chain, pose);
}

function lawcos_sss$1(aLen, bLen, cLen) {
  const v = (aLen ** 2 + bLen ** 2 - cLen ** 2) / (2 * aLen * bLen);
  return Math.acos(Math.min(1, Math.max(-1, v)));
}
function twoBoneSolver(tar, chain, pose, debug) {
  const lnk0 = chain.links[0];
  const lnk1 = chain.links[1];
  const root = pose.getBone(lnk0.index);
  const rot = new Quat().fromMul(tar.pworld.rot, root.local.rot);
  const bendAxis = Vec3.fromQuat(rot, lnk0.axes.ortho);
  let rad = lawcos_sss$1(lnk0.len, tar.dist, lnk1.len);
  rot.pmulAxisAngle(bendAxis, -rad).pmulInvert(tar.pworld.rot);
  pose.setLocalRot(lnk0.index, rot);
  const pRot = pose.getWorldRotation(lnk1.pindex);
  rad = Math.PI - lawcos_sss$1(lnk0.len, lnk1.len, tar.dist);
  rot.fromMul(pRot, lnk1.bind.rot).pmulAxisAngle(bendAxis, rad).pmulInvert(pRot);
  pose.setLocalRot(lnk1.index, rot);
}

function limbCompose(target, chain, pose, debug) {
  target.resolveTarget(chain, pose);
  lookSolver(target, chain, pose);
  if (target.dist >= chain.len)
    chain.resetPoseLocal(pose, 1);
  else
    twoBoneSolver(target, chain, pose);
}

function swingTwistChainSolver(tar, chain, pose, Debug) {
  const cMax = chain.links.length - 1;
  const ptran = new Transform();
  const ctran = new Transform();
  const tDir = new Vec3();
  const dir = new Vec3();
  const sRot = new Quat();
  const tRot = new Quat();
  let t;
  let lnk;
  for (let i = 0; i <= cMax; i++) {
    lnk = chain.links[i];
    t = i / cMax;
    pose.getWorldTransform(lnk.pindex, ptran);
    ctran.fromMul(ptran, lnk.bind);
    tDir.fromLerp(tar.swing, tar.altSwing, t).norm();
    dir.fromQuat(ctran.rot, lnk.axes.swing);
    sRot.fromSwing(dir, tDir).mul(ctran.rot);
    tDir.fromLerp(tar.twist, tar.altTwist, t).norm();
    dir.fromQuat(sRot, lnk.axes.twist);
    tRot.fromSwing(dir, tDir).mul(sRot).pmulInvert(ptran.rot);
    pose.setLocalRot(lnk.index, tRot);
  }
}

const IK_SOLVERS = {
  "root": rootCompose,
  "look": limbCompose$1,
  "limb": limbCompose,
  "swingchain": swingTwistChainSolver
};

class IKSet {
  // #region MAIN
  name = "";
  // Help to identify which limb to access
  order = 0;
  // Order this set should execute in
  target = new IKTarget();
  // Handles IK Target data solvers will use
  // target2  : IKTarget | null  = null;
  solver;
  // Any IK Solver
  chain;
  // IK Bone Chain
  constructor(name, order = 0) {
    this.name = name;
    this.order = order;
  }
  /** Set bones using the current TPose to store bind & axes information */
  setBones(bones, tPose, axes = BoneAxes.UFR) {
    this.chain = new IKChain(tPose.getBones(bones), axes);
    return this;
  }
  /** set which solver to be executed using the target onto the chain */
  setSolver(s) {
    this.solver = typeof s === "string" ? IK_SOLVERS[s] : s;
    return this;
  }
  // #endregion
  updatePose(pose, debug) {
    this.solver(this.target, this.chain, pose, debug);
  }
}
class IKRig {
  // #region MAIN
  sets = [];
  // Collection of Chained Bones
  names = {};
  // Names to Index Mapping
  pose;
  // Working Pose
  constructor(tpose) {
    this.pose = tpose.clone();
  }
  // #endregion
  // #region SETUP
  addSet(opt) {
    opt = Object.assign({
      order: 0,
      name: "",
      bones: [],
      axes: BoneAxes.UFR,
      solver: "look"
    }, opt);
    if (opt.bones.length === 0)
      return this;
    if (!opt.name)
      opt.name = "set" + this.sets.length;
    const s = new IKSet(opt.name, opt.order).setBones(opt.bones, this.pose, opt.axes).setSolver(opt.solver);
    this.sets.push(s);
    this.#reorder();
    return this;
  }
  /** Sort IKSet collection & update name-index mapping */
  #reorder() {
    this.sets.sort((a, b) => a.order === b.order ? 0 : a.order < b.order ? -1 : 1);
    for (let i = 0; i < this.sets.length; i++) {
      this.names[this.sets[i].name] = i;
    }
  }
  // #endregion
  // #region GETTERS / SETTERS
  getSet(name) {
    return this.sets[this.names[name]];
  }
  getEndPosition(name) {
    const s = this.sets[this.names[name]];
    return this.pose.getWorldPosition(s.chain.lastLink.index);
  }
  setTargetPositions(name, tarPos, polPos) {
    const s = this.sets[this.names[name]];
    if (s)
      s.target.setPositions(tarPos, polPos);
    else
      console.log("Setting target name not found", name);
    return this;
  }
  // #endregion
  // #region EXECUTION
  runSolvers(debug) {
    this.executor(this, debug);
    return this;
  }
  // Default Executor, Will use IKSet.order to determine execution order
  executor = (rig, debug) => {
    for (const s of this.sets)
      s.updatePose(this.pose, debug);
    this.pose.updateWorld();
  };
  // #endregion
}

function lawcos_sss(aLen, bLen, cLen) {
  const v = (aLen ** 2 + bLen ** 2 - cLen ** 2) / (2 * aLen * bLen);
  return Math.acos(Math.min(1, Math.max(-1, v)));
}
function ZSolver(tar, chain, pose, debug) {
  const l0 = chain.links[0];
  const l1 = chain.links[1];
  const l2 = chain.links[2];
  const aLen = l0.len;
  const bLen = l1.len;
  const cLen = l2.len;
  const bhLen = bLen * 0.5;
  const ratio = (aLen + bhLen) / (aLen + bLen + cLen);
  const taLen = tar.dist * ratio;
  const tbLen = tar.dist - taLen;
  const ptran = new Transform();
  const ctran = new Transform();
  const axis = new Vec3().fromCross(tar.twist, tar.swing);
  const rot = new Quat();
  let rad;
  const root = pose.getBone(l0.index)?.local ?? l0.bind;
  pose.getWorldTransform(l0.pindex, ptran);
  ctran.fromMul(ptran, root);
  rad = lawcos_sss(aLen, taLen, bhLen);
  rot.copy(ctran.rot).pmulAxisAngle(axis, -rad).pmulInvert(ptran.rot);
  pose.setLocalRot(l0.index, rot);
  pose.getWorldTransform(l1.pindex, ptran);
  ctran.fromMul(ptran, l1.bind);
  rad = Math.PI - lawcos_sss(aLen, bhLen, taLen);
  rot.copy(ctran.rot).pmulAxisAngle(axis, rad).pmulInvert(ptran.rot);
  pose.setLocalRot(l1.index, rot);
  pose.getWorldTransform(l2.pindex, ptran);
  ctran.fromMul(ptran, l2.bind);
  rad = Math.PI - lawcos_sss(cLen, bhLen, tbLen);
  rot.copy(ctran.rot).pmulAxisAngle(axis, -rad).pmulInvert(ptran.rot);
  pose.setLocalRot(l2.index, rot);
}

function zCompose(target, chain, pose, debug) {
  target.resolveTarget(chain, pose);
  lookSolver(target, chain, pose);
  if (target.dist >= chain.len)
    chain.resetPoseLocal(pose, 1);
  else
    ZSolver(target, chain, pose);
}

class Maths {
  // #region CONSTANTS
  static TAU = 6.283185307179586;
  // PI * 2
  static PI_H = 1.5707963267948966;
  static TAU_INV = 1 / 6.283185307179586;
  static PI_Q = 0.7853981633974483;
  static PI_Q3 = 1.5707963267948966 + 0.7853981633974483;
  static PI_270 = Math.PI + 1.5707963267948966;
  static DEG2RAD = 0.01745329251;
  // PI / 180
  static RAD2DEG = 57.2957795131;
  // 180 / PI
  static EPSILON = 1e-6;
  static PHI = 1.618033988749895;
  // Goldren Ratio, (1 + sqrt(5)) / 2
  //#endregion
  // #region OPERATIONS
  static clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }
  static clampGrad(v) {
    return Math.max(-1, Math.min(1, v));
  }
  static saturate(v) {
    return Math.max(0, Math.min(1, v));
  }
  static fract(f) {
    return f - Math.floor(f);
  }
  static nearZero(v) {
    return Math.abs(v) <= Maths.EPSILON ? 0 : v;
  }
  static dotToDeg(dot) {
    return Math.acos(Maths.clampGrad(dot)) * Maths.RAD2DEG;
  }
  static remap(x, xMin, xMax, zMin, zMax) {
    return (x - xMin) / (xMax - xMin) * (zMax - zMin) + zMin;
  }
  static snap(x, step) {
    return Math.floor(x / step) * step;
  }
  static norm(min, max, v) {
    return (v - min) / (max - min);
  }
  // Modulas that handles Negatives ex "Maths.mod( -1, 5 ) = 4
  static mod(a, b) {
    const v = a % b;
    return v < 0 ? b + v : v;
  }
  static lerp(a, b, t) {
    return a * (1 - t) + b * t;
  }
  // Logarithmic Interpolation
  static eerp(a, b, t) {
    return a * (b / a) ** t;
  }
  // Move value to the closest step
  static roundStep(value, step) {
    return Math.round(value / step) * step;
  }
  // https://docs.unity3d.com/ScriptReference/Mathf.SmoothDamp.html
  // https://github.com/Unity-Technologies/UnityCsReference/blob/a2bdfe9b3c4cd4476f44bf52f848063bfaf7b6b9/Runtime/Export/Math/Mathf.cs#L308
  static smoothDamp(cur, tar, vel, dt, smoothTime = 1e-4, maxSpeed = Infinity) {
    smoothTime = Math.max(1e-4, smoothTime);
    const omega = 2 / smoothTime;
    const x = omega * dt;
    const exp = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
    let change = cur - tar;
    const maxChange = maxSpeed * smoothTime;
    change = Math.min(maxChange, Math.max(change, -maxChange));
    const temp = (vel + omega * change) * dt;
    vel = (vel - omega * temp) * exp;
    let val = cur - change + (change + temp) * exp;
    if (tar - cur > 0 && val > tar) {
      val = tar;
      vel = 0;
    }
    return [val, vel];
  }
  // #endregion
}

class Heap {
  // #region MAIN
  items = [];
  compare;
  constructor(fnCompare) {
    this.compare = fnCompare;
  }
  // #endregion
  // #region GETTERS
  get length() {
    return this.items.length;
  }
  // #endregion
  // #region METHODS
  /** Add item to the heap which will then get stored into the correct spot */
  add(n) {
    const idx = this.items.length;
    this.items.push(n);
    if (idx !== 0)
      this.bubbleUp(idx);
    return this;
  }
  /** Remove item from heap, if no index is set then it pops off the first item */
  remove(idx = 0) {
    if (this.items.length === 0)
      return void 0;
    const i = this.items.length - 1;
    const rmItem = this.items[idx];
    const lastItem = this.items.pop();
    if (idx === i || this.items.length === 0 || lastItem === void 0)
      return rmItem;
    this.items[idx] = lastItem;
    this.bubbleDown(idx);
    return rmItem;
  }
  /** Pass in a task reference to find & remove it from heap */
  removeItem(itm) {
    const idx = this.items.indexOf(itm);
    if (idx !== -1) {
      this.remove(idx);
      return true;
    }
    return false;
  }
  // #endregion
  // #region SHIFTING
  /** Will move item down the tree by swopping the parent with one 
      of it's 2 children when conditions are met */
  bubbleDown(idx) {
    const ary = this.items;
    const len = ary.length;
    const itm = ary[idx];
    let lft = 0;
    let rit = 0;
    let mov = -1;
    while (idx < len) {
      lft = idx * 2 + 1;
      rit = idx * 2 + 2;
      mov = -1;
      if (lft < len && this.compare(ary[lft], itm))
        mov = lft;
      if (rit < len && this.compare(ary[rit], itm)) {
        if (mov === -1 || this.compare(ary[rit], ary[lft]))
          mov = rit;
      }
      if (mov === -1)
        break;
      [ary[idx], ary[mov]] = // Swop
      [ary[mov], ary[idx]];
      idx = mov;
    }
    return this;
  }
  /** Will move item up the tree by swopping with it's parent if conditions are met */
  bubbleUp(idx) {
    const ary = this.items;
    let pidx;
    while (idx > 0) {
      pidx = Math.floor((idx - 1) / 2);
      if (!this.compare(ary[idx], ary[pidx]))
        break;
      [ary[idx], ary[pidx]] = // Swop
      [ary[pidx], ary[idx]];
      idx = pidx;
    }
    return this;
  }
  // #endregion
}

class AnimationTask {
  remainingTime;
  elapsedTime;
  duration;
  onUpdate;
  constructor(durationSec, fnOnUpdate) {
    this.duration = durationSec;
    this.remainingTime = durationSec;
    this.elapsedTime = 0;
    this.onUpdate = fnOnUpdate;
  }
  get normTime() {
    return Math.min(1, Math.max(0, this.elapsedTime / this.duration));
  }
}
class AnimationQueue {
  // #region MAIN
  queue = new Heap((a, b) => a.remainingTime < b.remainingTime);
  // #endregion
  addTask(duration, fn) {
    this.enqueue(new AnimationTask(duration, fn));
    return this;
  }
  enqueue(task) {
    this.queue.add(task);
    return this;
  }
  // dequeue(){}
  update(dt) {
    if (this.queue.length === 0)
      return;
    for (const task of this.queue.items) {
      task.elapsedTime += dt;
      task.remainingTime = task.duration - task.elapsedTime;
      try {
        task.onUpdate(task);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("Error running an animation task:", msg);
      }
    }
    while (this.queue.length > 0 && this.queue.items[0].remainingTime <= 0) {
      this.queue.remove();
    }
  }
}

class Easing {
  //-----------------------------------------------
  static quadIn(k) {
    return k * k;
  }
  static quadOut(k) {
    return k * (2 - k);
  }
  static quadInOut(k) {
    if ((k *= 2) < 1)
      return 0.5 * k * k;
    return -0.5 * (--k * (k - 2) - 1);
  }
  //-----------------------------------------------
  static cubicIn(k) {
    return k * k * k;
  }
  static cubicOut(k) {
    return --k * k * k + 1;
  }
  static cubicInOut(k) {
    if ((k *= 2) < 1)
      return 0.5 * k * k * k;
    return 0.5 * ((k -= 2) * k * k + 2);
  }
  //-----------------------------------------------
  static quartIn(k) {
    return k * k * k * k;
  }
  static quartOut(k) {
    return 1 - --k * k * k * k;
  }
  static quartInOut(k) {
    if ((k *= 2) < 1)
      return 0.5 * k * k * k * k;
    return -0.5 * ((k -= 2) * k * k * k - 2);
  }
  //-----------------------------------------------
  static quintIn(k) {
    return k * k * k * k * k;
  }
  static quintOut(k) {
    return --k * k * k * k * k + 1;
  }
  static quintInOut(k) {
    if ((k *= 2) < 1)
      return 0.5 * k * k * k * k * k;
    return 0.5 * ((k -= 2) * k * k * k * k + 2);
  }
  //-----------------------------------------------
  static sineIn(k) {
    return 1 - Math.cos(k * Math.PI / 2);
  }
  static sineOut(k) {
    return Math.sin(k * Math.PI / 2);
  }
  static sineInOut(k) {
    return 0.5 * (1 - Math.cos(Math.PI * k));
  }
  //-----------------------------------------------
  static expIn(k) {
    return k === 0 ? 0 : Math.pow(1024, k - 1);
  }
  static expOut(k) {
    return k === 1 ? 1 : 1 - Math.pow(2, -10 * k);
  }
  static exp_nOut(k) {
    if (k === 0 || k === 1)
      return k;
    if ((k *= 2) < 1)
      return 0.5 * Math.pow(1024, k - 1);
    return 0.5 * (-Math.pow(2, -10 * (k - 1)) + 2);
  }
  //-----------------------------------------------
  static circIn(k) {
    return 1 - Math.sqrt(1 - k * k);
  }
  static circOut(k) {
    return Math.sqrt(1 - --k * k);
  }
  static circInOut(k) {
    if ((k *= 2) < 1)
      return -0.5 * (Math.sqrt(1 - k * k) - 1);
    return 0.5 * (Math.sqrt(1 - (k -= 2) * k) + 1);
  }
  //-----------------------------------------------
  static elasticIn(k) {
    if (k === 0 || k === 1)
      return k;
    return -Math.pow(2, 10 * (k - 1)) * Math.sin((k - 1.1) * 5 * Math.PI);
  }
  static elasticOut(k) {
    if (k === 0 || k === 1)
      return k;
    return Math.pow(2, -10 * k) * Math.sin((k - 0.1) * 5 * Math.PI) + 1;
  }
  static elasticInOut(k) {
    if (k === 0 || k === 1)
      return k;
    k *= 2;
    if (k < 1)
      return -0.5 * Math.pow(2, 10 * (k - 1)) * Math.sin((k - 1.1) * 5 * Math.PI);
    return 0.5 * Math.pow(2, -10 * (k - 1)) * Math.sin((k - 1.1) * 5 * Math.PI) + 1;
  }
  //-----------------------------------------------
  static backIn(k) {
    return k * k * ((1.70158 + 1) * k - 1.70158);
  }
  static backOut(k) {
    return --k * k * ((1.70158 + 1) * k + 1.70158) + 1;
  }
  static backInOut(k) {
    const s = 1.70158 * 1.525;
    if ((k *= 2) < 1)
      return 0.5 * (k * k * ((s + 1) * k - s));
    return 0.5 * ((k -= 2) * k * ((s + 1) * k + s) + 2);
  }
  //-----------------------------------------------
  static bounceIn(k) {
    return 1 - Easing.bounceOut(1 - k);
  }
  static bounceOut(k) {
    if (k < 1 / 2.75)
      return 7.5625 * k * k;
    else if (k < 2 / 2.75)
      return 7.5625 * (k -= 1.5 / 2.75) * k + 0.75;
    else if (k < 2.5 / 2.75)
      return 7.5625 * (k -= 2.25 / 2.75) * k + 0.9375;
    else
      return 7.5625 * (k -= 2.625 / 2.75) * k + 0.984375;
  }
  static bounce_InOut(k) {
    if (k < 0.5)
      return Easing.bounceIn(k * 2) * 0.5;
    return Easing.bounceOut(k * 2 - 1) * 0.5 + 0.5;
  }
  //-----------------------------------------------
  // EXTRAS
  static smoothTStep(t) {
    return t * t * (3 - 2 * t);
  }
  static sigmoid(t, k = 0) {
    return (t - k * t) / (k - 2 * k * Math.abs(t) + 1);
  }
  static bellCurve(t) {
    return (Math.sin(2 * Math.PI * (t - 0.25)) + 1) * 0.5;
  }
  /** a = 1.5, 2, 4, 9 */
  static betaDistCurve(t, a) {
    return 4 ** a * (t * (1 - t)) ** a;
  }
  static bouncy(t, jump = 6, offset = 1) {
    const rad = 6.283185307179586 * t;
    return (offset + Math.sin(rad)) / 2 * Math.sin(jump * rad);
  }
  /** bounce ease out */
  static bounce(t) {
    return (Math.sin(t * Math.PI * (0.2 + 2.5 * t * t * t)) * Math.pow(1 - t, 2.2) + t) * (1 + 1.2 * (1 - t));
  }
}

const EventType = {
  Frame: 0,
  Time: 1
};
const LerpType = {
  Step: 0,
  Linear: 1,
  Cubic: 2
};

class QuatBuffer {
  // #region MAIN
  buf;
  result = new Quat();
  constructor(buf) {
    this.buf = buf;
  }
  // #endregion
  // #region GETTERS
  get(i, out = this.result) {
    i *= 4;
    out[0] = this.buf[i + 0];
    out[1] = this.buf[i + 1];
    out[2] = this.buf[i + 2];
    out[3] = this.buf[i + 3];
    return out;
  }
  // #endregion
  // #region INTERPOLATION
  nblend(ai, bi, t, out = this.result) {
    ai *= 4;
    bi *= 4;
    const ary = this.buf;
    const a_x = ary[ai + 0];
    const a_y = ary[ai + 1];
    const a_z = ary[ai + 2];
    const a_w = ary[ai + 3];
    const b_x = ary[bi + 0];
    const b_y = ary[bi + 1];
    const b_z = ary[bi + 2];
    const b_w = ary[bi + 3];
    const dot = a_x * b_x + a_y * b_y + a_z * b_z + a_w * b_w;
    const ti = 1 - t;
    const s = dot < 0 ? -1 : 1;
    out[0] = ti * a_x + t * b_x * s;
    out[1] = ti * a_y + t * b_y * s;
    out[2] = ti * a_z + t * b_z * s;
    out[3] = ti * a_w + t * b_w * s;
    return out.norm();
  }
  slerp(ai, bi, t, out = this.result) {
    ai *= 4;
    bi *= 4;
    const ary = this.buf;
    const ax = ary[ai + 0], ay = ary[ai + 1], az = ary[ai + 2], aw = ary[ai + 3];
    let bx = ary[bi + 0], by = ary[bi + 1], bz = ary[bi + 2], bw = ary[bi + 3];
    let omega, cosom, sinom, scale0, scale1;
    cosom = ax * bx + ay * by + az * bz + aw * bw;
    if (cosom < 0) {
      cosom = -cosom;
      bx = -bx;
      by = -by;
      bz = -bz;
      bw = -bw;
    }
    if (1 - cosom > 1e-6) {
      omega = Math.acos(cosom);
      sinom = Math.sin(omega);
      scale0 = Math.sin((1 - t) * omega) / sinom;
      scale1 = Math.sin(t * omega) / sinom;
    } else {
      scale0 = 1 - t;
      scale1 = t;
    }
    out[0] = scale0 * ax + scale1 * bx;
    out[1] = scale0 * ay + scale1 * by;
    out[2] = scale0 * az + scale1 * bz;
    out[3] = scale0 * aw + scale1 * bw;
    return out;
  }
  // #endregion
}

class TrackQuat {
  // #region MAIN
  boneIndex = -1;
  // Bine index in skeleton this track will animate
  timeIndex = -1;
  // Which timestamp array it uses.
  lerpType = LerpType.Linear;
  values;
  // Flat data of animation
  vbuf;
  // Quat wrapper over flat data
  constructor(lerpType = LerpType.Linear) {
    this.lerpType = lerpType;
  }
  // #endregion
  // #region SETTERS
  setData(data) {
    this.values = new Float32Array(data);
    this.vbuf = new QuatBuffer(this.values);
    return this;
  }
  // #endregion
  // #region METHODS
  apply(pose, fi) {
    switch (this.lerpType) {
      case LerpType.Step:
        pose.setLocalRot(this.boneIndex, this.vbuf.get(fi.kB));
        break;
      case LerpType.Linear:
        this.vbuf.nblend(fi.kB, fi.kC, fi.t);
        pose.setLocalRot(this.boneIndex, this.vbuf.result);
        break;
      default:
        console.log("QuatTrack - unknown lerp type");
        break;
    }
    return this;
  }
  // #endregion
}

class Vec3Buffer {
  // #region MAIN
  buf;
  result = new Vec3();
  constructor(buf) {
    this.buf = buf;
  }
  // #endregion
  // #region GETTERS
  get(i, out = this.result) {
    i *= 3;
    out[0] = this.buf[i + 0];
    out[1] = this.buf[i + 1];
    out[2] = this.buf[i + 2];
    return out;
  }
  // #endregion
  // #region INTERPOLATION
  lerp(ai, bi, t, out = this.result) {
    const ary = this.buf;
    const ti = 1 - t;
    ai *= 3;
    bi *= 3;
    out[0] = ti * ary[ai + 0] + t * ary[bi + 0];
    out[1] = ti * ary[ai + 1] + t * ary[bi + 1];
    out[2] = ti * ary[ai + 2] + t * ary[bi + 2];
    return out;
  }
  // #endregion
}

class TrackVec3 {
  // #region MAIN
  boneIndex = -1;
  // Bine index in skeleton this track will animate
  timeIndex = -1;
  // Which timestamp array it uses.
  lerpType = LerpType.Linear;
  values;
  // Flat data of animation
  vbuf;
  // Vec3 wrapper over flat data
  constructor(lerpType = LerpType.Linear) {
    this.lerpType = lerpType;
  }
  // #endregion
  // #region SETTERS
  setData(data) {
    this.values = new Float32Array(data);
    this.vbuf = new Vec3Buffer(this.values);
    return this;
  }
  // #endregion
  // #region METHODS
  apply(pose, fi) {
    switch (this.lerpType) {
      case LerpType.Step:
        pose.setLocalPos(this.boneIndex, this.vbuf.get(fi.kB));
        break;
      case LerpType.Linear:
        pose.setLocalPos(this.boneIndex, this.vbuf.lerp(fi.kB, fi.kC, fi.t));
        break;
      default:
        console.log("Vec3Track - unknown lerp type", this.lerpType);
        break;
    }
    return this;
  }
  // #endregion
}

class AnimationEvent {
  name = "";
  type = EventType.Frame;
  start = -1;
  // Starting Frame or Time
  duration = -1;
  // How many frames or seconds this event lasts
  constructor(name, start = 0, eventType = EventType.Frame, duration = -1) {
    this.name = name;
    this.start = start;
    this.duration = duration;
    this.type = eventType;
  }
}

class RootMotion {
  // #region MAIN
  values;
  // Flat array of positions for each frame
  vbuf;
  //
  frameCount = 0;
  // How many frames worth of data exists
  timeStampIdx = -1;
  // Which time stamp to be used by root motion
  p0 = [0, 0, 0];
  // Preallocate vec objects so no need to reallocated every frame.
  p1 = [0, 0, 0];
  result = [0, 0, 0];
  constructor(data) {
    this.values = data;
    this.vbuf = new Vec3Buffer(this.values);
    this.frameCount = data.length / 3;
  }
  // #endregion
  getBetweenFrames(f0, t0, f1, t1) {
    const p0 = this.p0;
    const p1 = this.p1;
    const rtn = this.result;
    if (f0 > f1) {
      if (f0 + 1 < this.frameCount) {
        this.vbuf.get(this.frameCount - 1, p1);
        this.vbuf.lerp(f0, f0 + 1, t0, p0);
        p0[0] = p1[0] - p0[0];
        p0[1] = p1[1] - p0[1];
        p0[2] = p1[2] - p0[2];
      } else {
        p0[0] = 0;
        p0[1] = 0;
        p0[2] = 0;
      }
      this.vbuf.lerp(f1, f1 + 1, t1, p1);
      rtn[0] = p0[0] + p1[0];
      rtn[1] = p0[1] + p1[1];
      rtn[2] = p0[2] + p1[2];
      return rtn;
    }
    this.vbuf.lerp(f0, f0 + 1, t0, p0);
    if (f1 + 1 < this.frameCount)
      this.vbuf.lerp(f1, f1 + 1, t1, p1);
    else
      this.vbuf.get(f1, p1);
    rtn[0] = p1[0] - p0[0];
    rtn[1] = p1[1] - p0[1];
    rtn[2] = p1[2] - p0[2];
    return rtn;
  }
}

class Clip {
  // #region MAIN
  name = "";
  // Clip Name
  frameCount = 0;
  // Total frames in animation
  duration = 0;
  // Total animation time
  timeStamps = [];
  // Different sets of shared time stamps
  tracks = [];
  // Collection of animations broke out as Rotation, Position & Scale
  events = void 0;
  // Collection of animation events
  rootMotion = void 0;
  // Root motion for this animation
  isLooped = true;
  // Is the animation to run in a loop
  constructor(name = "") {
    this.name = name;
  }
  // #endregion
  // #region EVENTS
  addEvent(name, start, eventType = EventType.Frame, duration = -1) {
    if (!this.events)
      this.events = [];
    this.events.push(new AnimationEvent(name, start, eventType, duration));
    return this;
  }
  setRootMotionData(data) {
    const rm = new RootMotion(data);
    for (let i = 0; i < this.timeStamps.length; i++) {
      if (this.timeStamps[i].length === rm.frameCount) {
        rm.timeStampIdx = i;
        break;
      }
    }
    this.rootMotion = rm;
    return this;
  }
  // #endregion
  // #region METHODS
  timeAtFrame(f) {
    if (f >= 0 && f < this.frameCount) {
      for (const ts of this.timeStamps) {
        if (ts.length === this.frameCount)
          return ts[f];
      }
    }
    return -1;
  }
  // #endregion
  // #region DEBUG
  debugInfo(arm) {
    const pose = arm?.bindPose;
    const lerpKeys = Object.keys(LerpType);
    const getLerpName = (v) => lerpKeys.find((k) => LerpType[k] === v);
    let bName = "";
    let trackType = "";
    console.log(
      "Clip Name [ %s ] 	 Track Count [ %d ] 	 Max frames [ %d ]",
      this.name,
      this.tracks.length,
      this.frameCount
    );
    for (const t of this.tracks) {
      if (pose)
        bName = pose.bones[t.boneIndex].name;
      if (t instanceof TrackQuat)
        trackType = "quat";
      else if (t instanceof TrackVec3)
        trackType = "vec3";
      else
        trackType = "Unknown";
      console.log(
        "Bone [ %s ] 	 Type [ %s ] 	 Lerp Type [ %s ] 	 Frames [ %d ]",
        bName,
        trackType,
        getLerpName(t.lerpType),
        this.timeStamps[t.timeIndex].length
      );
    }
  }
  // #endregion
}

class PoseAnimator {
  // #region MAIN
  isRunning = false;
  clip = void 0;
  // Animation Clip
  clock = 0;
  // Animation Clock
  fInfo = [];
  // Clips can have multiple Timestamps
  scale = 1;
  // Scale the speed of the animation
  onEvent = void 0;
  //
  eventCache = void 0;
  placementMask = [0, 1, 0];
  // Used when inPlace is turned on. Set what to reset.
  // #endregion
  // #region SETTERS
  setClip(clip) {
    this.clip = clip;
    this.clock = 0;
    this.fInfo.length = 0;
    for (let i = 0; i < clip.timeStamps.length; i++) {
      this.fInfo.push(new FrameInfo());
    }
    if (clip.events && !this.eventCache) {
      this.eventCache = /* @__PURE__ */ new Map();
    }
    this.computeFrameInfo();
    return this;
  }
  setScale(s) {
    this.scale = s;
    return this;
  }
  // #endregion
  // #region FRAME CONTROLS
  step(dt) {
    if (this.clip && this.isRunning) {
      const tick = dt * this.scale;
      if (!this.clip.isLooped && this.clock + tick >= this.clip.duration) {
        this.clock = this.clip.duration;
        this.isRunning = false;
      } else {
        if (this.clock + tick >= this.clip.duration) {
          this.eventCache?.clear();
        }
        this.clock = (this.clock + tick) % this.clip.duration;
      }
      this.computeFrameInfo();
      if (this.clip.events && this.onEvent) {
        this.checkEvents();
      }
    }
    return this;
  }
  atTime(t) {
    if (this.clip) {
      this.clock = t % this.clip.duration;
      this.computeFrameInfo();
    }
    return this;
  }
  atFrame(n) {
    if (!this.clip)
      return this;
    n = Math.max(0, Math.min(this.clip.frameCount, n));
    const tsAry = this.clip.timeStamps;
    const fiAry = this.fInfo;
    let tsLen;
    let ts;
    let fi;
    for (let i = 0; i < tsAry.length; i++) {
      ts = tsAry[i];
      fi = fiAry[i];
      tsLen = ts.length - 1;
      fi.t = 0;
      fi.kA = n <= tsLen ? n : tsLen;
      fi.kB = fi.kA;
      fi.kC = fi.kA;
      fi.kD = fi.kA;
    }
    return this;
  }
  // #endregion
  // #region METHODS
  start() {
    this.isRunning = true;
    return this;
  }
  stop() {
    this.isRunning = false;
    return this;
  }
  usePlacementReset(mask = [0, 1, 0]) {
    this.placementMask = mask;
    return this;
  }
  updatePose(pose) {
    if (this.clip) {
      let t;
      for (t of this.clip.tracks) {
        t.apply(pose, this.fInfo[t.timeIndex]);
      }
    }
    if (this.placementMask) {
      pose.bones[0].local.pos.mul(this.placementMask);
    }
    pose.updateWorld();
    return this;
  }
  getMotion() {
    const rm = this?.clip?.rootMotion;
    if (rm) {
      const fi = this.fInfo[rm.timeStampIdx];
      return rm.getBetweenFrames(fi.pkB, fi.pt, fi.kB, fi.t);
    }
    return null;
  }
  // #endregion
  // #region INTERNAL METHODS
  computeFrameInfo() {
    if (!this.clip)
      return;
    const time = this.clock;
    let fi;
    let ts;
    let imin;
    let imax;
    let imid;
    for (let i = 0; i < this.fInfo.length; i++) {
      fi = this.fInfo[i];
      if (this.clip.timeStamps[i].length === 0) {
        fi.singleFrame();
        continue;
      }
      ts = this.clip.timeStamps[i];
      fi.pkB = Math.max(fi.kB, 0);
      fi.pt = fi.t;
      if (time < ts[fi.kB] || time > ts[fi.kC] || fi.kB === -1) {
        imin = 0;
        imax = ts.length - 1;
        while (imin < imax) {
          imid = imin + imax >>> 1;
          if (time < ts[imid])
            imax = imid;
          else
            imin = imid + 1;
        }
        if (imax <= 0) {
          fi.kB = 0;
          fi.kC = 1;
        } else {
          fi.kB = imax - 1;
          fi.kC = imax;
        }
        fi.kA = Maths.mod(fi.kB - 1, ts.length);
        fi.kD = Maths.mod(fi.kC + 1, ts.length);
      }
      fi.t = (time - ts[fi.kB]) / (ts[fi.kC] - ts[fi.kB]);
    }
  }
  checkEvents() {
    if (!this?.clip?.events || !this.onEvent)
      return;
    for (const fi of this.fInfo) {
      for (const evt of this.clip.events) {
        if (evt.start >= fi.pkB && evt.start < fi.kB && !this.eventCache?.get(evt.name)) {
          this.eventCache?.set(evt.name, true);
          try {
            this.onEvent(evt.name);
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            console.error("Error while calling animation event callback:", msg);
          }
          break;
        }
      }
    }
  }
  // #endregion
}
class FrameInfo {
  t = 0;
  // Lerp Time
  kA = -1;
  // Keyframe Pre Tangent
  kB = -1;
  // Keyframe Lerp Start
  kC = -1;
  // Keyframe Lerp End
  kD = -1;
  // Keyframe Post Tangent
  pkB = 0;
  // Previous Lerp Start
  pt = 0;
  // Previous Lerp Time
  // Set info for single frame timeStamp
  singleFrame() {
    this.t = 1;
    this.kA = 0;
    this.kB = -1;
    this.kC = -1;
    this.kD = 0;
    this.pkB = 0;
    this.pt = 0;
  }
}

class BoneLink {
  // #region MAIN
  srcIndex = -1;
  // Bone index in source tpose
  tarIndex = -1;
  // Bone index in target tpose
  qSrcParent = new Quat();
  // Cache the bone's parent worldspace quat
  qDotCheck = new Quat();
  // Cache the src bone worldspace quat for DOT Check
  qSrcToTar = new Quat();
  // Handles transformation from Src WS to Tar WS
  qTarParent = new Quat();
  // Cache tpose parent ws rotation, to make it easy to transform Tar WS to Tar LS
  constructor(srcIdx, tarIdx) {
    if (srcIdx != null)
      this.srcIndex = srcIdx;
    if (tarIdx != null)
      this.tarIndex = tarIdx;
  }
  // #endregion
  // #region METHODS
  fromBones(src, tar) {
    this.srcIndex = src.index;
    this.tarIndex = tar.index;
    return this;
  }
  bind(src, tar) {
    const srcBone = src.bones[this.srcIndex];
    const tarBone = tar.bones[this.tarIndex];
    this.qDotCheck.copy(srcBone.world.rot);
    this.qSrcParent.copy(
      srcBone.pindex !== -1 ? src.bones[srcBone.pindex].world.rot : src.offset.rot
    );
    this.qTarParent.fromInvert(
      tarBone.pindex !== -1 ? tar.bones[tarBone.pindex].world.rot : tar.offset.rot
    );
    this.qSrcToTar.fromInvert(srcBone.world.rot).mul(tarBone.world.rot);
    return this;
  }
  // #endregion
}
class Retarget {
  // #region MAIN
  animator = new PoseAnimator();
  links = /* @__PURE__ */ new Map();
  srcPose;
  // Starting "Pose" for both should be very similar for retargeting to work
  tarPose;
  // ... a TPose is the most ideal pose for retargeting
  srcHip = new Vec3();
  // TODO, Need to handle Root bone too
  tarHip = new Vec3();
  hipScale = 1;
  // #endregion
  // #region METHODS
  /** Set the tpose for both skeletons */
  useTPoses(src, tar) {
    this.srcPose = src.clone();
    this.tarPose = tar.clone();
    return this;
  }
  bindBone(srcName, tarName) {
    const bSrc = this.srcPose.getBone(srcName);
    const bTar = this.tarPose.getBone(tarName);
    if (!bSrc || !bTar) {
      console.log("Can not link bones", srcName, tarName);
      return this;
    }
    const lnk = new BoneLink().fromBones(bSrc, bTar).bind(this.srcPose, this.tarPose);
    this.links.set(window.crypto.randomUUID(), lnk);
    return this;
  }
  bindBatch(ary) {
    const pSrc = this.srcPose;
    const pTar = this.tarPose;
    let bSrc;
    let bTar;
    let lnk;
    let lnkName;
    for (const i of ary) {
      bSrc = pSrc.getBone(i.src);
      bTar = pTar.getBone(i.tar);
      if (!bSrc || !bTar) {
        console.log("Can not link bones", i);
        continue;
      }
      lnk = new BoneLink().fromBones(bSrc, bTar).bind(this.srcPose, this.tarPose);
      if (i.incPos === true) {
        this.srcHip.copy(bSrc.world.pos).nearZero();
        this.tarHip.copy(bTar.world.pos).nearZero();
        this.hipScale = Math.abs(this.srcHip[1] / this.tarHip[1]);
        lnkName = "hip";
      } else {
        lnkName = window.crypto.randomUUID();
      }
      this.links.set(lnkName, lnk);
    }
    return this;
  }
  autoBindTPoses(src, tar) {
    this.srcPose = src.clone();
    this.tarPose = tar.clone();
    const srcBonemap = new BoneMap(src);
    const tarBonemap = new BoneMap(tar);
    let tarBone;
    for (const [key, srcBone] of srcBonemap.bones) {
      tarBone = tarBonemap.bones.get(key);
      if (!tarBone)
        continue;
      if (tarBone.isChain || srcBone.isChain) {
        const srcMax = srcBone.count - 1;
        const tarMax = tarBone.count - 1;
        this.links.set(
          key + "_first",
          new BoneLink(srcBone.index, tarBone.index).bind(src, tar)
        );
        this.links.set(
          key + "_last",
          new BoneLink(
            srcBone.items[srcMax].index,
            tarBone.items[tarMax].index
          ).bind(src, tar)
        );
        for (let i = 1; i <= Math.min(srcMax - 1, tarMax - 1); i++) {
          this.links.set(
            key + "_" + i,
            new BoneLink(
              srcBone.items[i].index,
              tarBone.items[i].index
            ).bind(src, tar)
          );
        }
      } else {
        this.links.set(
          key,
          new BoneLink(srcBone.index, tarBone.index).bind(src, tar)
        );
      }
    }
    const hip = this.links.get("hip");
    if (hip) {
      const srcBone = src.bones[hip.srcIndex];
      const tarBone2 = tar.bones[hip.tarIndex];
      this.srcHip.copy(srcBone.world.pos).nearZero();
      this.tarHip.copy(tarBone2.world.pos).nearZero();
      this.hipScale = Math.abs(this.srcHip[1] / this.tarHip[1]);
    }
    return this;
  }
  // #endregion
  // #region CONTROL ANIMATION
  step(dt) {
    this.animator.step(dt).updatePose(this.srcPose);
    this.srcPose.updateWorld();
    this.retargetPose();
    this.tarPose.updateWorld();
    return this;
  }
  // #endregion
  // #region CALCULATIONS
  retargetPose() {
    const diff = new Quat();
    const tmp = new Quat();
    let lnk;
    let srcBone;
    let tarBone;
    for (lnk of this.links.values()) {
      srcBone = this.srcPose.bones[lnk.srcIndex];
      tarBone = this.tarPose.bones[lnk.tarIndex];
      diff.fromMul(lnk.qSrcParent, srcBone.local.rot);
      diff.mul(
        Quat.dot(diff, lnk.qDotCheck) < 0 ? tmp.fromNegate(lnk.qSrcToTar) : lnk.qSrcToTar
        // No correction needed, transform to target tpose
      );
      tarBone.local.rot.fromMul(lnk.qTarParent, diff);
    }
    lnk = this.links.get("hip");
    if (lnk) {
      srcBone = this.srcPose.bones[lnk.srcIndex];
      tarBone = this.tarPose.bones[lnk.tarIndex];
      tarBone.local.pos.fromSub(srcBone.world.pos, this.srcHip).scale(this.hipScale).add(this.tarHip);
    }
  }
  // #endregion
}

export { AnimationQueue, Armature, Bone, BoneAxes, BoneBindings, BoneMap, BoneSockets, Clip, DQTSkin$1 as DQTSkin, DQTSkin as DualQuatSkin, Easing, IKChain, IKLink, IKRig, IKTarget, LerpType, Mat4, Maths, MatrixSkin, Pose, PoseAnimator, Quat, Retarget, RootMotion, SQTSkin, TrackQuat, TrackVec3, TranMatrixSkin, Transform, Vec3, deltaMoveSolver, limbCompose, limbCompose$1 as lookCompose, lookSolver, rootCompose, swingTwistChainSolver, twoBoneSolver, zCompose };
