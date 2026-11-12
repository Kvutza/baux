<?xml version="1.0"?>
<!-- SPDX-License-Identifier: CC0-1.0 -->
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:fr="http://www.jonmsterling.com/jms-005P.xml"
  xmlns:mml="http://www.w3.org/1998/Math/MathML"
  xmlns:html="http://www.w3.org/1999/xhtml">

  <!-- The following ensures that node not matched by a template will show up as an error. -->
  <xsl:template match="node()|@*">
    <xsl:copy>
      <span style="background-color:red">
        <xsl:text>[</xsl:text>
        <xsl:value-of select="name(.)" />
        <xsl:text>]</xsl:text>
      </span>
      <span style="background-color:red">
        <xsl:apply-templates select="node()|@*" />
      </span>
    </xsl:copy>
  </xsl:template>

  <!-- HTML and MathML nodes should be copied with namespace prefixes stripped.-->
  <xsl:template match="html:*">
    <xsl:element namespace="http://www.w3.org/1999/xhtml" name="{local-name()}">
      <xsl:apply-templates select="@* | node()" />
    </xsl:element>
  </xsl:template>

  <xsl:template match="html:a">
    <xsl:variable name="check-href"
      select="translate(normalize-space(@href), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ&#9;&#10;&#13; ', 'abcdefghijklmnopqrstuvwxyz')" />
    <xsl:variable name="unsafe-scheme"
      select="starts-with($check-href, 'javascript:') or starts-with($check-href, 'vbscript:') or starts-with($check-href, 'data:') or starts-with($check-href, 'blob:') or starts-with($check-href, 'file:') or starts-with($check-href, 'http:') or starts-with($check-href, '//')" />
    <xsl:element namespace="http://www.w3.org/1999/xhtml" name="a">
      <xsl:if test="@href">
        <xsl:attribute name="href">
          <xsl:choose>
            <xsl:when test="$unsafe-scheme">#</xsl:when>
            <xsl:otherwise><xsl:value-of select="normalize-space(@href)" /></xsl:otherwise>
          </xsl:choose>
        </xsl:attribute>
      </xsl:if>
      <xsl:if test="not($unsafe-scheme) and (starts-with($check-href, 'https://') or @target = '_blank')">
        <xsl:if test="@target = '_blank'">
          <xsl:attribute name="target">_blank</xsl:attribute>
        </xsl:if>
        <xsl:attribute name="rel">noopener noreferrer</xsl:attribute>
      </xsl:if>
      <xsl:apply-templates select="@*[local-name() != 'href' and local-name() != 'target' and local-name() != 'rel']" />
      <xsl:apply-templates />
    </xsl:element>
  </xsl:template>

  <xsl:template match="div">
    <div>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </div>
  </xsl:template>

  <xsl:template match="section">
    <section>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </section>
  </xsl:template>

  <xsl:template match="span">
    <span>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </span>
  </xsl:template>

  <xsl:template match="dl">
    <dl>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </dl>
  </xsl:template>

  <xsl:template match="dt">
    <dt>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </dt>
  </xsl:template>

  <xsl:template match="dd">
    <dd>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </dd>
  </xsl:template>

  <xsl:template match="p">
    <p>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </p>
  </xsl:template>

  <xsl:template match="pre">
    <pre>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </pre>
  </xsl:template>

  <xsl:template match="code">
    <code>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </code>
  </xsl:template>

  <xsl:template match="ul">
    <ul>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </ul>
  </xsl:template>

  <xsl:template match="ol">
    <ol>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </ol>
  </xsl:template>

  <xsl:template match="li">
    <li>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </li>
  </xsl:template>

  <xsl:template match="strong">
    <strong>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </strong>
  </xsl:template>

  <xsl:template match="em">
    <em>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </em>
  </xsl:template>

  <xsl:template match="a">
    <xsl:variable name="check-href"
      select="translate(normalize-space(@href), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ&#9;&#10;&#13; ', 'abcdefghijklmnopqrstuvwxyz')" />
    <xsl:variable name="unsafe-scheme"
      select="starts-with($check-href, 'javascript:') or starts-with($check-href, 'vbscript:') or starts-with($check-href, 'data:') or starts-with($check-href, 'blob:') or starts-with($check-href, 'file:') or starts-with($check-href, 'http:') or starts-with($check-href, '//')" />
    <a>
      <xsl:if test="@href">
        <xsl:attribute name="href">
          <xsl:choose>
            <xsl:when test="$unsafe-scheme">#</xsl:when>
            <xsl:otherwise><xsl:value-of select="normalize-space(@href)" /></xsl:otherwise>
          </xsl:choose>
        </xsl:attribute>
      </xsl:if>
      <xsl:if test="not($unsafe-scheme) and (starts-with($check-href, 'https://') or @target = '_blank')">
        <xsl:if test="@target = '_blank'">
          <xsl:attribute name="target">_blank</xsl:attribute>
        </xsl:if>
        <xsl:attribute name="rel">noopener noreferrer</xsl:attribute>
      </xsl:if>
      <xsl:apply-templates select="@*[local-name() != 'href' and local-name() != 'target' and local-name() != 'rel']" />
      <xsl:apply-templates />
    </a>
  </xsl:template>

  <xsl:template match="table">
    <table>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </table>
  </xsl:template>

  <xsl:template match="thead">
    <thead>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </thead>
  </xsl:template>

  <xsl:template match="tbody">
    <tbody>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </tbody>
  </xsl:template>

  <xsl:template match="tr">
    <tr>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </tr>
  </xsl:template>

  <xsl:template match="th">
    <th>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </th>
  </xsl:template>

  <xsl:template match="td">
    <td>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </td>
  </xsl:template>

  <xsl:template match="figure">
    <figure>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </figure>
  </xsl:template>

  <xsl:template match="figcaption">
    <figcaption>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </figcaption>
  </xsl:template>

  <xsl:template match="button">
    <button>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </button>
  </xsl:template>

  <xsl:template match="sup">
    <sup>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </sup>
  </xsl:template>

  <xsl:template match="sub">
    <sub>
      <xsl:apply-templates select="@*" />
      <xsl:apply-templates />
    </sub>
  </xsl:template>

  <xsl:template match="svg | g | path | polyline | polygon | line | rect | circle | text | tspan | defs | clipPath | title">
    <xsl:element namespace="http://www.w3.org/2000/svg" name="{local-name()}">
      <xsl:apply-templates select="@* | node()" />
    </xsl:element>
  </xsl:template>

  <xsl:template match="@id">
    <xsl:attribute name="id"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@class">
    <xsl:attribute name="class"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-symbol">
    <xsl:attribute name="data-symbol"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-kind">
    <xsl:attribute name="data-kind"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-module">
    <xsl:attribute name="data-module"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@scope">
    <xsl:attribute name="scope"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@aria-label">
    <xsl:attribute name="aria-label"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@type">
    <xsl:attribute name="type"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@hidden">
    <xsl:attribute name="hidden"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@role">
    <xsl:attribute name="role"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@aria-live">
    <xsl:attribute name="aria-live"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@aria-hidden">
    <xsl:attribute name="aria-hidden"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@aria-pressed">
    <xsl:attribute name="aria-pressed"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@aria-selected">
    <xsl:attribute name="aria-selected"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@aria-atomic">
    <xsl:attribute name="aria-atomic"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@focusable">
    <xsl:attribute name="focusable"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@tabindex">
    <xsl:attribute name="tabindex"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@title">
    <xsl:attribute name="title"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-bench-fig">
    <xsl:attribute name="data-bench-fig"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-regime">
    <xsl:attribute name="data-regime"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-view">
    <xsl:attribute name="data-view"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-scrub">
    <xsl:attribute name="data-scrub"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-seed">
    <xsl:attribute name="data-seed"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-arm">
    <xsl:attribute name="data-arm"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-delta">
    <xsl:attribute name="data-delta"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-value">
    <xsl:attribute name="data-value"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-summary">
    <xsl:attribute name="data-summary"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-x-left">
    <xsl:attribute name="data-x-left"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-x-right">
    <xsl:attribute name="data-x-right"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-log-min">
    <xsl:attribute name="data-log-min"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@data-log-max">
    <xsl:attribute name="data-log-max"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@viewBox">
    <xsl:attribute name="viewBox"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@width">
    <xsl:attribute name="width"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@height">
    <xsl:attribute name="height"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@x">
    <xsl:attribute name="x"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@y">
    <xsl:attribute name="y"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@dx">
    <xsl:attribute name="dx"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@dy">
    <xsl:attribute name="dy"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@x1">
    <xsl:attribute name="x1"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@y1">
    <xsl:attribute name="y1"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@x2">
    <xsl:attribute name="x2"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@y2">
    <xsl:attribute name="y2"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@cx">
    <xsl:attribute name="cx"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@cy">
    <xsl:attribute name="cy"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@r">
    <xsl:attribute name="r"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@rx">
    <xsl:attribute name="rx"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@ry">
    <xsl:attribute name="ry"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@d">
    <xsl:attribute name="d"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@points">
    <xsl:attribute name="points"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@fill">
    <xsl:attribute name="fill"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@fill-opacity">
    <xsl:attribute name="fill-opacity"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@stroke">
    <xsl:attribute name="stroke"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@stroke-width">
    <xsl:attribute name="stroke-width"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@stroke-dasharray">
    <xsl:attribute name="stroke-dasharray"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@stroke-linecap">
    <xsl:attribute name="stroke-linecap"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@stroke-linejoin">
    <xsl:attribute name="stroke-linejoin"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@stroke-opacity">
    <xsl:attribute name="stroke-opacity"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@opacity">
    <xsl:attribute name="opacity"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@text-anchor">
    <xsl:attribute name="text-anchor"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@dominant-baseline">
    <xsl:attribute name="dominant-baseline"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@font-size">
    <xsl:attribute name="font-size"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@font-weight">
    <xsl:attribute name="font-weight"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@font-family">
    <xsl:attribute name="font-family"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@font-style">
    <xsl:attribute name="font-style"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@letter-spacing">
    <xsl:attribute name="letter-spacing"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@transform">
    <xsl:attribute name="transform"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@clip-path">
    <xsl:attribute name="clip-path"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@clipPathUnits">
    <xsl:attribute name="clipPathUnits"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@vector-effect">
    <xsl:attribute name="vector-effect"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@paint-order">
    <xsl:attribute name="paint-order"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@pointer-events">
    <xsl:attribute name="pointer-events"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="@shape-rendering">
    <xsl:attribute name="shape-rendering"><xsl:value-of select="." /></xsl:attribute>
  </xsl:template>

  <xsl:template match="mml:*">
    <xsl:element namespace="http://www.w3.org/1998/Math/MathML" name="{local-name()}">
      <xsl:apply-templates select="@* | node()" />
    </xsl:element>
  </xsl:template>

  <xsl:template match="fr:figure">
    <figure>
      <xsl:apply-templates />
    </figure>
  </xsl:template>

  <xsl:template match="fr:figcaption">
    <figcaption>
      <xsl:apply-templates />
    </figcaption>
  </xsl:template>

  <xsl:template match="fr:p">
    <p>
      <xsl:apply-templates />
    </p>
  </xsl:template>

  <xsl:template match="fr:code">
    <code>
      <xsl:apply-templates />
    </code>
  </xsl:template>

  <xsl:template match="fr:pre">
    <pre>
      <xsl:apply-templates />
    </pre>
  </xsl:template>

  <xsl:template match="fr:em">
    <em>
      <xsl:apply-templates />
    </em>
  </xsl:template>

  <xsl:template match="fr:strong">
    <strong>
      <xsl:apply-templates />
    </strong>
  </xsl:template>

  <xsl:template match="fr:ol">
    <ol>
      <xsl:apply-templates />
    </ol>
  </xsl:template>

  <xsl:template match="fr:ul">
    <ul>
      <xsl:apply-templates />
    </ul>
  </xsl:template>

  <xsl:template match="fr:li">
    <li>
      <xsl:apply-templates />
    </li>
  </xsl:template>

  <xsl:template match="fr:blockquote">
    <blockquote>
      <xsl:apply-templates />
    </blockquote>
  </xsl:template>

  <xsl:template match="fr:img[@src]">
    <img src="{@src}" />
  </xsl:template>

  <xsl:template match="fr:error | fr:info">
    <span class="error">
      <xsl:apply-templates />
    </span>
  </xsl:template>

  <xsl:template match="fr:info">
    <span class="info">
      <xsl:apply-templates />
    </span>
  </xsl:template>

  <xsl:template match="fr:tex[@display='block']">
    <xsl:text>\[</xsl:text>
    <xsl:value-of select="." />
    <xsl:text>\]</xsl:text>
  </xsl:template>

  <xsl:template match="fr:tex[not(@display='block')]">
    <xsl:text>\(</xsl:text>
    <xsl:value-of select="." />
    <xsl:text>\)</xsl:text>
  </xsl:template>

</xsl:stylesheet>
