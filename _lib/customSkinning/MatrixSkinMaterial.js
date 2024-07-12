import * as THREE from '@three';

export default function MatrixSkinMaterial( val='cyan', skin ){
    const isTex    = ( val instanceof THREE.Texture || val.isTexture );
    const uniforms = {
        pose : { value: skin?.offsetBuffer },
    };

    if( !isTex ){
        let color;
        switch( typeof val ){
            case 'string':
            case 'number': color = new THREE.Color( val ); break;
            case 'object': if( Array.isArray( val ) ) color = new THREE.Color( val[0], val[1], val[2] ); break;
            default: color = new THREE.Color( 'red' ); break;
        }

        uniforms.color     = { type: 'vec3', value: color };
    }else{
        uniforms.texBase   = { type: 'sampler2D', value: val };
    }

    const matConfig = {
        side            : THREE.DoubleSide,
        uniforms        : uniforms,
        vertexShader    : VERT_SRC,
        fragmentShader	: ( !isTex )? FRAG_COL : FRAG_TEX,
    }

    const mat       = new THREE.RawShaderMaterial( matConfig );
    mat.extensions  = { derivatives : true }; // If not using WebGL2.0 and Want to use dfdx or fwidth, Need to load extension

    Object.defineProperty( mat, 'map', {
        set( c ){ mat.uniforms.texBase.value = c ; },
    });

    return mat;
}

// #region SHADER CODE

// HANDLE SKINNING
const VERT_SRC = `#version 300 es
in vec3 position;   // Vertex Position
in vec3 normal;     // Vertex Normal
in vec2 uv;         // Vertex Texcoord
in vec4 skinWeight; // Bone Weights
in vec4 skinIndex;  // Bone Indices

#define MAXBONES 110             // Arrays can not be dynamic, so must set a size
uniform mat4 pose[ MAXBONES ];

uniform mat4 modelMatrix;       // Matrices should be filled in by THREE.JS Automatically.
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec3 frag_wpos;             // Fragment World Space Position
out vec3 frag_norm;             // Fragment Normal
out vec2 frag_uv;               // Fragment Texcoord

////////////////////////////////////////////////////////////////////////

mat4 getBoneMatrix( mat4[ MAXBONES ] pose, vec4 idx, vec4 wgt ){
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NORMALIZE BONE WEIGHT VECTOR - INCASE MODEL WASN'T PREPARED LIKE THAT
    If Weights are not normalized, Merging the Bone Offsets will create artifacts */
    int a = int( idx.x ),
        b = int( idx.y ),
        c = int( idx.z ),
        d = int( idx.w );
    
    wgt *= 1.0 / ( wgt.x + wgt.y + wgt.z + wgt.w );

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // MERGE THE BONE OFFSETS BASED ON WEIGHT
    mat4 bone_wgt =
        pose[ a ] * wgt.x +  
        pose[ b ] * wgt.y +
        pose[ c ] * wgt.z +
        pose[ d ] * wgt.w;

    return bone_wgt;
}

////////////////////////////////////////////////////////////////////////

void main() {
    mat4 boneMatrix = getBoneMatrix( pose, skinIndex, skinWeight );         // Get the Skinning Matrix
    mat4 mbMatrix   = modelMatrix * boneMatrix;                             // Merge Model and Bone Matrices together

    vec4 wpos       = mbMatrix * vec4( position, 1.0 );                     // Use new Matrix to Transform Vertices
    frag_wpos       = wpos.xyz;                                             // Save WorldSpace Position for Fragment Shader
    frag_norm       = mat3( transpose( inverse( mbMatrix ) ) ) * normal;    // Transform Normals using bone + model matrix
    frag_uv         = uv;

    gl_Position     = projectionMatrix * viewMatrix * wpos;
    //gl_Position     = projectionMatrix * viewMatrix * vec4( position, 1.0 );
}`;

// FRAGMENT THAT HANDLES BASE COLOR & LIGHTING
const FRAG_COL = `#version 300 es
precision mediump float;

////////////////////////////////////////////////////////////////////////

out     vec4 out_color;
in      vec3 frag_wpos;
in      vec3 frag_norm;

uniform vec3 color;

////////////////////////////////////////////////////////////////////////

#define LITCNT 2
const vec3[] light_pos = vec3[](
    vec3( 0.0, 0.5, 1.0 ),
    vec3( -1.0, 0.0, 1.0 )
);

float computePointLights( vec3[LITCNT] lights, vec3 norm ){
    vec3 light_vec;
    vec3 light_dir;

    float dist;
    float attenuation;
    float diffuse     = 0.0;
    float constant    = 0.5;
    float linear      = 0.5;
    float quadratic   = 0.5;
    
    for( int i=0; i < LITCNT; i++ ){
        light_vec       = lights[i].xyz - frag_wpos;
        light_dir       = normalize( light_vec );
        dist            = length( light_vec );
        attenuation     = 1.0 / ( constant + linear * dist + quadratic * (dist * dist) );
        diffuse        += max( dot( norm, light_dir ), 0.0 ) * attenuation;
    }

    return diffuse;
}

const vec3 sun_pos = vec3( 10.0, 10.0, 10.0 );
float computeDirLight( vec3 lPos, vec3 norm ){
    return max( dot( norm, normalize( lPos ) ), 0.0 );
}

void main(){
    //vec3 norm   = normalize( cross( dFdx(frag_wpos), dFdy(frag_wpos) ) ); // Low Poly Normals
    vec3 norm     = normalize( frag_norm ); // Model's Normals            
    
    float diffuse = 0.15; // ambient light
    diffuse      += computePointLights( light_pos, norm );
    diffuse      += computeDirLight( sun_pos, norm );

    out_color     = vec4( color * diffuse, 1.0 );

    //out_color.rgb = vec3( 1.0, 0.0, 0.0 );
}`;

// FRAGMENT THAT ONLY RENDERS TEXTURE
const FRAG_TEX = `#version 300 es
precision mediump float;

////////////////////////////////////////////////////////////////////////

out     vec4 out_color;
in      vec2 frag_uv;

uniform sampler2D texBase;

////////////////////////////////////////////////////////////////////////

void main(){
    out_color = texture( texBase, frag_uv );
}`;

// #endregion